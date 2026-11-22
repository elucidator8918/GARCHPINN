"""Device handling without a GPU: static audit of tensor factories, device selection and
GPU assignment with a mocked GPU count, a meta-device forward pass; real CUDA tests are
skipped unless CUDA is present (they run on the Linux GPU box)."""
import ast
import math
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float64)

from garchpinn import utils as U  # noqa: E402
from garchpinn.config import Config  # noqa: E402
from garchpinn.losses import total_loss  # noqa: E402
from garchpinn.models import build_model  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
SOURCES = sorted((ROOT / "garchpinn").rglob("*.py")) + [ROOT / "main.py", ROOT / "ablate.py"]
FACTORIES = {"zeros", "ones", "empty", "full", "tensor", "as_tensor", "arange", "linspace",
             "randn", "rand", "randint", "eye", "Generator", "from_numpy"}
CUDA = torch.cuda.is_available()


def test_every_tensor_factory_has_a_device():
    """Each torch.<factory>(...) passes device= or is marked '# cpu-ok' (module init,
    deliberate CPU work, or a call that keeps its input tensor's device)."""
    bad = []
    for f in SOURCES:
        src = f.read_text()
        lines = src.splitlines()
        for node in ast.walk(ast.parse(src)):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name) and node.func.value.id == "torch"
                    and node.func.attr in FACTORIES):
                continue
            if any(k.arg == "device" for k in node.keywords):
                continue
            span = lines[node.lineno - 1: node.end_lineno]
            if not any("# cpu-ok" in l for l in span):
                bad.append(f"{f.relative_to(ROOT)}:{node.lineno}: torch.{node.func.attr}")
    assert not bad, "\n".join(bad)


def test_numpy_conversions_go_through_cpu():
    bad = []
    for f in SOURCES:
        for i, l in enumerate(f.read_text().splitlines(), 1):
            if re.search(r"(?<!cpu\(\))\.numpy\(\)", l):
                bad.append(f"{f.relative_to(ROOT)}:{i}: {l.strip()}")
    assert not bad, "\n".join(bad)


def test_no_mps():
    with pytest.raises(ValueError):
        U.resolve_device("mps")
    for f in SOURCES:
        assert '"mps"' not in f.read_text().replace('startswith("mps")', ""), f


def _mock_gpus(monkeypatch, n):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: n > 0)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: n)


def test_resolve_device_auto(monkeypatch):
    narrow = Config()
    wide = Config(streams=4096, chunk_len=128)
    assert U.resolve_device("cpu", wide) == "cpu"
    _mock_gpus(monkeypatch, 0)
    assert U.resolve_device("auto", wide) == "cpu"
    with pytest.raises(RuntimeError):
        U.resolve_device("cuda")
    _mock_gpus(monkeypatch, 3)
    assert U.resolve_device("auto", narrow) == "cpu"  # launch-bound: stay on CPU
    assert U.resolve_device("auto", wide) == "cuda:0"
    assert U.resolve_device("cuda", gpus="2,1") == "cuda:2"
    assert U.resolve_device("cuda:1") == "cuda:1"
    with pytest.raises(ValueError):
        U.resolve_device("cuda:3")


def test_assign_devices_one_gpu(monkeypatch):
    _mock_gpus(monkeypatch, 1)
    with pytest.warns(UserWarning, match="share one GPU"):
        assert U.assign_devices(4, "cuda") == ["cuda:0"] * 4
    assert U.assign_devices(3, "cpu") == ["cpu"] * 3


def test_assign_devices_three_gpus(monkeypatch):
    _mock_gpus(monkeypatch, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert U.assign_devices(6, "cuda") == ["cuda:0", "cuda:1", "cuda:2"] * 2
        assert U.assign_devices(2, "cuda", gpus="0,1,2") == ["cuda:0", "cuda:1"]
    assert U.assign_devices(4, "cuda", gpus="2,0") == ["cuda:2", "cuda:0"] * 2
    with pytest.raises(ValueError):
        U.visible_gpus("0,3")
    assert U.visible_gpus() == [0, 1, 2]


@pytest.mark.parametrize("model", ["garch", "egarch"])
def test_forward_and_loss_on_meta(model):
    """Training-mode forward + loss on the meta device: any CPU tensor created inside the
    recursion or the loss would raise a device mismatch."""
    cfg = Config(model=model)
    m = build_model(cfg, 5).to("meta")
    r = torch.empty(4, 30, device="meta")
    lx, ex = torch.empty(4, 30, device="meta"), torch.empty(4, 30, 5, device="meta")
    out, st = m(r, lx, ex)
    out2, _ = m(r, lx, ex, st)
    loss, _ = total_loss(m, out2, torch.ones(4, 30, device="meta"), cfg, lx)
    assert loss.device.type == "meta" and out["log_h"].device.type == "meta"


def test_save_load_roundtrip(tmp_path):
    m = build_model(Config(), 5)
    U.save_model(m, str(tmp_path / "m"))
    m2 = U.load_model(build_model(Config(), 5), str(tmp_path / "m"))
    for a, b in zip(m.state_dict().values(), m2.state_dict().values()):
        assert torch.equal(a, b)


# ------------------------------------------------------------------ real CUDA (GPU box only)
def _small_run(device, tmp_path):
    from test_ablation import _fake_csv
    from garchpinn.ablation import make_config, run_one
    from garchpinn.data import get_series
    csv = tmp_path / "fake.csv"
    if not csv.exists():
        _fake_csv(csv)
    cfg = make_config("baseline", "garch", "daily", 0,
                      {"data_path": str(csv), "stage1_iters": 30, "n_paths": 100, "epochs": 2})
    cfg.device = device
    row, _, losses = run_one(cfg, get_series(cfg))
    return row, losses


@pytest.mark.skipif(not CUDA, reason="needs CUDA")
def test_cuda_matches_cpu(tmp_path):
    rc, lc = _small_run("cpu", tmp_path)
    rg, lg = _small_run("cuda:0", tmp_path)
    for k, v in rc.items():
        if isinstance(v, float) and k != "runtime_s" and math.isfinite(v):
            assert abs(v - rg[k]) <= 1e-8 * max(1.0, abs(v)), (k, v, rg[k])
    np.testing.assert_allclose(lc, lg, rtol=1e-5, equal_nan=True)  # stored as float32


@pytest.mark.skipif(not CUDA or torch.cuda.device_count() < 2, reason="needs >= 2 GPUs")
def test_two_jobs_on_two_gpus(tmp_path):
    import ablate
    from test_ablation import _fake_csv
    _fake_csv(tmp_path / "fake.csv")
    df = ablate.main(["--data", str(tmp_path / "fake.csv"), "--quick", "--variants", "baseline",
                      "--models", "garch", "--seeds", "2", "--epochs", "1", "--jobs", "2",
                      "--device", "cuda", "--gpus", "0,1", "--out", str(tmp_path / "abl")])
    assert set(df.device) == {"cuda:0", "cuda:1"}
