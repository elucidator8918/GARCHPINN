"""Seeding, model persistence and device selection."""
import json
import os
import random
import warnings

import numpy as np
import torch


def set_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_default_dtype(torch.float64)


def _jsonable(v):
    if isinstance(v, torch.Tensor):
        return v.detach().cpu().tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, dict):
        return {k: _jsonable(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_jsonable(x) for x in v]
    return v


def save_model(model, path: str, cfg=None) -> None:
    """Save state_dict to <path>.pt and paper-notation params (+config) to <path>.json."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, path + ".pt")
    meta = {"params": _jsonable(model.params()) if hasattr(model, "params") else {}}
    if cfg is not None:
        meta["config"] = _jsonable(cfg.to_dict())
    with open(path + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def load_model(model, path: str, map_location=None):
    """Load <path>.pt into an already-built model of the same architecture (onto the
    model's device unless map_location is given)."""
    p = path if path.endswith(".pt") else path + ".pt"
    if map_location is None:
        map_location = next(iter(model.state_dict().values())).device
    model.load_state_dict(torch.load(p, map_location=map_location))
    return model


# ------------------------------------------------------------------ devices
# auto -> CUDA only for wide workloads: the recursion is sequential (one small kernel per
# time step), so a GPU only pays off with many parallel streams or Monte Carlo paths.
AUTO_CUDA_MIN_WIDTH = 1 << 18  # streams * chunk_len
AUTO_CUDA_MIN_PATHS = 200_000  # n_paths


def visible_gpus(gpus=None):
    """GPU indices to use (relative to CUDA_VISIBLE_DEVICES, as torch sees them).
    gpus: None/"all" -> every visible GPU; "0,2" or a list -> those (validated)."""
    n = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if gpus is None or gpus == "all" or gpus == "":
        return list(range(n))
    ids = [int(g) for g in (gpus.split(",") if isinstance(gpus, str) else gpus) if str(g).strip() != ""]
    bad = [g for g in ids if not 0 <= g < n]
    if bad:
        raise ValueError(f"GPUs {bad} not visible (torch sees {n}; CUDA_VISIBLE_DEVICES="
                         f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r})")
    return ids


def is_wide(cfg):
    return (cfg.streams * cfg.chunk_len >= AUTO_CUDA_MIN_WIDTH
            or cfg.stage1_streams * 128 >= AUTO_CUDA_MIN_WIDTH or cfg.n_paths >= AUTO_CUDA_MIN_PATHS)


def resolve_device(spec="auto", cfg=None, gpus=None):
    """'cpu' | 'cuda' | 'cuda:K' | 'auto' -> a concrete device string. MPS is never used
    (no float64). 'cuda' picks the first of `gpus`; auto picks it only if cfg is wide."""
    spec = (spec or "auto").lower()
    if spec.startswith("mps"):
        raise ValueError("MPS has no float64 support; use cpu or cuda")
    if spec == "cpu":
        return "cpu"
    if spec == "auto":
        if not torch.cuda.is_available() or cfg is None or not is_wide(cfg):
            return "cpu"
        spec = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError(f"--device {spec} requested but CUDA is not available")
    if spec == "cuda":
        ids = visible_gpus(gpus)
        return f"cuda:{ids[0]}"
    if spec.startswith("cuda:"):
        visible_gpus([int(spec.split(":")[1])])  # validate
        return spec
    raise ValueError(f"unknown device {spec!r}")


def assign_devices(n_jobs, spec="cpu", gpus=None, cfg=None):
    """Device for each of n_jobs worker slots. cuda: round-robin over the GPUs (several
    workers per GPU when n_jobs > #GPUs); cuda:K: all on K."""
    spec = (spec or "auto").lower()
    if spec == "auto":
        spec = "cuda" if resolve_device("auto", cfg, gpus) != "cpu" else "cpu"
    if spec == "cpu":
        return ["cpu"] * n_jobs
    if spec.startswith("cuda:"):
        dev = resolve_device(spec)
        if n_jobs > 1:
            warnings.warn(f"{n_jobs} workers share {dev}")
        return [dev] * n_jobs
    resolve_device("cuda", gpus=gpus)
    ids = visible_gpus(gpus)
    if n_jobs > len(ids) and len(ids) == 1:
        warnings.warn(f"{n_jobs} workers share one GPU (cuda:{ids[0]})")
    return [f"cuda:{ids[i % len(ids)]}" for i in range(n_jobs)]


def pin_device(device):
    """Call in a worker process: make `device` the current CUDA device."""
    if str(device).startswith("cuda"):
        torch.cuda.set_device(torch.device(device))


def make_generator(seed, device="cpu"):
    """torch.Generator on `device` (CPU and CUDA streams differ for the same seed)."""
    return torch.Generator(device=device).manual_seed(int(seed))
