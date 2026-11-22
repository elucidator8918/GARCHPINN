"""Quick end-to-end ablation run on synthetic 5-min data (2 variants x 1 seed)."""
import os

import numpy as np
import pandas as pd
import torch

torch.set_default_dtype(torch.float64)

import ablate  # noqa: E402
from garchpinn.ablation import holm  # noqa: E402


def _fake_csv(path, n_days=160, seed=0):
    rng = np.random.default_rng(seed)
    rows, p, lv = [], 6500.0, np.log(1e-6)
    for d in pd.bdate_range("2015-01-01", periods=n_days):
        lv = 0.97 * lv + 0.03 * np.log(1e-6) + 0.2 * rng.standard_normal()
        for k in range(103):
            s = np.exp(0.5 * lv)
            o = p
            p = p * np.exp(s * rng.standard_normal())
            rows.append((".FTSE", "Market Price", "Intraday 5Min", d + pd.Timedelta(hours=8, minutes=5 * k), 0,
                         o, max(o, p) * (1 + 0.3 * s), min(o, p) * (1 - 0.3 * s), p, 100))
    pd.DataFrame(rows, columns=["#RIC", "Domain", "Type", "Date-Time", "GMT Offset", "Open", "High", "Low",
                                "Last", "No. Trades"]).to_csv(path, index=False)


def test_holm():
    adj = holm([0.01, 0.04, 0.03, np.nan])
    assert np.allclose(adj[:3], [0.03, 0.06, 0.06]) and np.isnan(adj[3])


def test_quick_ablation(tmp_path):
    data = tmp_path / "fake.csv"
    _fake_csv(data)
    out = tmp_path / "abl"
    args = ["--data", str(data), "--quick", "--variants", "baseline,tau0", "--models", "garch",
            "--seeds", "1", "--epochs", "2", "--out", str(out)]
    df = ablate.main(args)
    assert len(df) == 2 and set(df.variant) == {"baseline", "tau0"}
    for c in ("test_l_r", "test_l_rx", "qlike_h0", "mse_h2", "val_nll", "p_persistence", "runtime_s"):
        assert c in df and df[c].notna().all(), c
    assert (df[df.variant == "tau0"].p_tau1 == 0).all()
    for f in ("results.csv", "results.json", "summary.md", "P1_bars.png", "P2_lambda_grid.png",
              "P3_learning_curves.png", "P4_dm_h0.png", "P5_param_shifts.png"):
        assert (out / f).exists(), f
    L = np.load(out / "losses" / df.losses_file.iloc[0])
    assert L.dtype == np.float32 and L.shape[0] == 3
    # resumable: a second call runs nothing new
    df2 = ablate.main(args)
    assert len(df2) == 2 and len(open(out / "runs.jsonl").read().splitlines()) == 2
    assert not os.path.exists(tmp_path / "results")
