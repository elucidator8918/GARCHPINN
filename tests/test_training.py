"""Stage-2 efficacy and NN identification (orthogonality) tests (float64)."""
import time

import numpy as np
import pandas as pd
import torch

torch.set_default_dtype(torch.float64)

from garchpinn.config import Config  # noqa: E402
from garchpinn.data import build_dataset  # noqa: E402
from garchpinn.losses import _lag_regressors, moment_residuals, orth_penalty, total_loss  # noqa: E402
from garchpinn.models import EGARCHPINN, GARCHPINN  # noqa: E402
from garchpinn.train import fit  # noqa: E402


def _sim_nonlinear(T=2000, amp=0.3, seed=0):
    """Realized GARCH plus a term it cannot represent: a day-of-week effect and a threshold
    1{z_t < -1} (centred) in log h_{t+1}."""
    rng = np.random.default_rng(seed)
    lh, r, x, cal = 0.0, [], [], []
    for t in range(T + 500):
        z, u = rng.standard_normal(), 0.4 * rng.standard_normal()
        lx = lh - 0.07 * z + 0.07 * (z * z - 1) + u
        r.append(np.exp(lh / 2) * z), x.append(np.exp(lx)), cal.append((t % 5) / 4)
        ang = 2 * np.pi * ((t + 1) % 5) / 5
        lh = 0.55 * lh + 0.4 * lx + amp * np.sin(ang) + amp * ((z < -1.0) - 0.1587)
    return pd.DataFrame({"r": r[500:], "x": x[500:], "cal": cal[500:]})


def test_stage2_beats_stage1_on_nonlinear_simulation():
    cfg = Config(epochs=40, hidden_dim=16, streams=4, econ_warmup=5, diagnostics=False)
    torch.manual_seed(0)
    ds = build_dataset(_sim_nonlinear(), cfg)
    m = GARCHPINN(n_exo=ds.exo.shape[1], hidden_dim=cfg.hidden_dim, nn_scale=cfg.nn_scale)
    t0 = time.time()
    h = fit(m, ds, cfg)
    assert time.time() - t0 < 60
    assert h["best_epoch"] > 0
    assert h["best_val_nll"] < h["stage1_val_nll"] - 0.01, (h["stage1_val_nll"], h["best_val_nll"])


def test_econ_params_frozen_during_warmup():
    cfg = Config(epochs=2, econ_warmup=2, hidden_dim=8, chunk_len=256, patience=5)
    ds = build_dataset(_sim_nonlinear(T=600), cfg)
    m = GARCHPINN(n_exo=ds.exo.shape[1], hidden_dim=8)
    from garchpinn.train import stage1_qmle, stage2_pinn
    stage1_qmle(m, ds, cfg)
    before = [p.detach().clone() for p in m.econ_parameters()]
    cfg.patience = 10**6  # keep the last state even if val does not improve
    stage2_pinn(m, ds, cfg)
    assert all(torch.equal(a, b) for a, b in zip(before, m.econ_parameters()))


def _out(B=2, L=50, seed=0):
    g = torch.Generator().manual_seed(seed)
    rn = lambda: torch.randn(B, L, generator=g)
    return {"log_h": rn(), "z": rn(), "u": rn(), "g": 0.1 * rn()}, rn()


def test_orth_regressors_by_variant():
    out, lx = _out()
    assert len(_lag_regressors(GARCHPINN(use_nn=False), out, lx)) == 2  # log h, log x
    assert len(_lag_regressors(GARCHPINN(use_nn=False), out, None)) == 1
    assert len(_lag_regressors(EGARCHPINN(use_nn=False), out, lx)) == 4  # log h, z, z^2-1, u


def test_orth_penalises_g_spanning_log_x():
    out, lx = _out()
    m = GARCHPINN(use_nn=False)
    base = float(orth_penalty(m, out, None, lx))
    out2 = dict(out)
    out2["g"] = out["g"].clone()
    out2["g"][:, 1:] = out2["g"][:, 1:] + 0.1 * lx[:, :-1]  # g_t loads on log x_{t-1}
    assert float(orth_penalty(m, out2, None, lx)) > base + 5e-3
    assert float(orth_penalty(m, out2, None, None)) < base + 1e-3  # invisible without log x


def test_orth_egarch_detects_u_and_z():
    out, _ = _out()
    m = EGARCHPINN(use_nn=False)
    base = float(orth_penalty(m, out, None))
    for k in ("u", "z"):
        o = dict(out)
        o["g"] = out["g"].clone()
        o["g"][:, 1:] = o["g"][:, 1:] + 0.1 * out[k][:, :-1]
        assert float(orth_penalty(m, o, None)) > base + 5e-3


def test_total_loss_threads_log_x_and_standardized_moments():
    out, lx = _out(B=1, L=400)
    m = GARCHPINN(n_exo=0, use_nn=True)
    cfg = Config()
    _, p = total_loss(m, out, None, cfg, lx)
    assert "orth" in p and "moment" in p
    # standardised R_moment is ~ chi2_k / n under the null and scale-free in u
    r1 = moment_residuals(out["z"], out["u"], 1.0, standardize=True)
    r2 = moment_residuals(out["z"], 3 * out["u"], 3.0, standardize=True)
    assert torch.allclose(r1, r2)
    assert float(r1) < 0.2
