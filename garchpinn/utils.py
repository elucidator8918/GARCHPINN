"""Seeding and model persistence."""
import json
import os
import random

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
    torch.save(model.state_dict(), path + ".pt")
    meta = {"params": _jsonable(model.params()) if hasattr(model, "params") else {}}
    if cfg is not None:
        meta["config"] = _jsonable(cfg.to_dict())
    with open(path + ".json", "w") as f:
        json.dump(meta, f, indent=2)


def load_model(model, path: str, map_location="cpu"):
    """Load <path>.pt into an already-built model of the same architecture."""
    p = path if path.endswith(".pt") else path + ".pt"
    model.load_state_dict(torch.load(p, map_location=map_location))
    return model
