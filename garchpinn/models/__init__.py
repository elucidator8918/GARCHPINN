from .base import RealizedPINN, psi
from .egarch_pinn import EGARCHPINN
from .garch_pinn import GARCHPINN

MODEL_CLASSES = {"garch": GARCHPINN, "egarch": EGARCHPINN}


def uses_overnight(cfg):
    """True when the data carry (on, r_on) exo columns (data.has_overnight) for this config."""
    if cfg.freq == "5min":
        return getattr(cfg, "overnight", "drop") == "model"
    return bool(getattr(cfg, "daily_close_to_close", False))


def build_model(cfg, n_exo=0):
    """Instantiate GARCHPINN / EGARCHPINN from a Config."""
    m = MODEL_CLASSES[cfg.model](n_exo=n_exo, hidden_dim=cfg.hidden_dim, use_nn=cfg.use_nn,
                                 nn_state_inputs=cfg.nn_state_inputs, nn_scale=cfg.nn_scale,
                                 log_h_bound=cfg.log_h_bound, overnight=uses_overnight(cfg))
    if getattr(cfg, "no_leverage", False):
        m.freeze_leverage()
    return m


__all__ = ["RealizedPINN", "GARCHPINN", "EGARCHPINN", "MODEL_CLASSES", "build_model", "uses_overnight", "psi"]
