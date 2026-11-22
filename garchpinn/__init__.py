"""GARCHPINN: physics-informed Realized GARCH / Realized EGARCH volatility models."""
import torch

torch.set_default_dtype(torch.float64)

from .config import Config  # noqa: E402
from .models import EGARCHPINN, GARCHPINN, RealizedPINN, build_model  # noqa: E402

__all__ = ["Config", "GARCHPINN", "EGARCHPINN", "RealizedPINN", "build_model"]
