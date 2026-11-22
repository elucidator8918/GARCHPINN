"""Regression tests for the econometrics review fixes (float64)."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float64)

from garchpinn.data import Dataset  # noqa: E402
from garchpinn.evaluate import dm_test  # noqa: E402
from garchpinn.losses import moment_residuals  # noqa: E402
from garchpinn.models import EGARCHPINN, GARCHPINN  # noqa: E402
from garchpinn.train import stage1_qmle  # noqa: E402

HIGH = {  # Realized GARCH with E log h = 4.5 (5-min returns in basis points), pi = 0.99
    "garch": dict(mu=0.0, omega=0.045, beta=0.584, gamma=0.40, xi=0.0, phi=1.015,
                  tau1=-0.07, tau2=0.07, sigma_u=0.4),
    "egarch": dict(mu=0.0, omega=0.045, beta=0.99, gamma=0.30, tau1=-0.10, tau2=0.04,
                   xi=0.0, phi=1.0, delta1=-0.07, delta2=0.07, sigma_u=0.4),
}
CLS = {"garch": GARCHPINN, "egarch": EGARCHPINN}


def test_guard_is_identity_on_data_range():
    m = GARCHPINN(use_nn=False)
    lh = torch.linspace(-8, 8, 101)
    assert torch.allclose(m._bound(lh), lh, atol=1e-3)
    assert float((m._bound(lh) - lh).abs()[40:61].max()) < 1e-5  # |lh| <= 1.6
    m.lh_center.fill_(4.5)
    lh2 = lh + 4.5
    assert torch.allclose(m._bound(lh2), lh2, atol=1e-3)
    assert float(m._bound(torch.tensor(1e4))) < 4.5 + 15 + 1  # still bounded far out


@pytest.mark.parametrize("v", ["garch", "egarch"])
def test_forecast_matches_mc_at_high_log_h(v):
    """Critical fix: with log h ~ 4.5 and a = 0.99 the closed form must match the bounded
    simulation of the same filter (the old 15 tanh(./15) gave ratios of 3 to 9)."""
    m = CLS[v](use_nn=False).load_params(**HIGH[v])
    r, lx, lh = m.simulate(800, seed=5)
    assert 3.5 < float(lh.mean()) < 5.5
    with torch.no_grad():
        out, state = m(r, lx)
    H = 102
    fc = m.forecast(out["log_h_next"], H)
    mc = m.simulate_forecast(state, H, n_paths=200_000, generator=torch.Generator().manual_seed(7))
    for k in (1, 12, 102):
        for key in ("h", "x"):
            a, b = float(fc[key][k - 1]), float(mc[key][k - 1])
            assert abs(a / b - 1) < 0.03, (v, key, k, a, b)


def test_qmle_not_distorted_at_high_log_h():
    """Critical fix: stage-1 QMLE at E log h = 4.3 recovers pi (old bound pushed it to 0.9999)."""
    truth = GARCHPINN(use_nn=False).load_params(
        mu=0.0, omega=0.0215, persistence=0.995, gamma=0.41, xi=0.0, phi=1.0,
        tau1=-0.07, tau2=0.07, sigma_u=0.375)
    r, lx, _ = truth.simulate(4000, seed=11)
    ds = Dataset(r, lx, r.new_zeros(4000, 0), None, n_train=4000, n_val=0)
    m = GARCHPINN(use_nn=False).load_params(**truth.params())
    stage1_qmle(m, ds, SimpleNamespace(stage1_streams=1, stage1_burn_in=22, stage1_iters=300))
    p = m.params()
    assert abs(p["persistence"] - 0.995) < 0.006, p
    assert abs(p["gamma"] - 0.41) < 0.1, p
    assert p["persistence"] < 0.9995


def test_horizon_one_needs_no_state_mgf():
    m = EGARCHPINN(use_nn=False).load_params(**{**HIGH["egarch"], "tau2": 10.0})
    fc = m.forecast(torch.tensor([0.1]), 1)  # psi(a^0) would be undefined; not needed at k=1
    assert torch.isfinite(fc["x"]).all()
    with pytest.raises(ValueError):
        m.forecast(torch.tensor([0.1]), 2)


def test_moment_penalty_does_not_inflate_sigma_u():
    gen = torch.Generator().manual_seed(0)
    n, rho = 4000, 0.3
    e = torch.randn(n, generator=gen)
    u = torch.zeros(n)
    for t in range(1, n):
        u[t] = rho * u[t - 1] + e[t]
    z = torch.randn(n, generator=gen)
    s = torch.tensor(1.3, requires_grad=True)
    pen = moment_residuals(z, u, s)
    assert pen > 1e-2  # autocorrelated u: the penalty is non-zero ...
    assert not pen.requires_grad  # ... but cannot be reduced through sigma_u


def test_dm_size_overlapping():
    rng = np.random.default_rng(0)
    k, n, rej = 22, 900, 0
    for _ in range(300):
        e = rng.standard_normal(n + k)
        d = np.convolve(e, np.ones(k), "valid")[:n]
        rej += dm_test(d, np.zeros(n), k=k)["pvalue"] < 0.05
    assert rej / 300 < 0.09, rej / 300


def test_stability_diagnostics():
    g = GARCHPINN(use_nn=False).load_params(**HIGH["garch"]).params()
    assert g["filter_stable"]
    e = EGARCHPINN(use_nn=False).load_params(**HIGH["egarch"]).params()
    assert e["e_log_abs_A"] < 0 and e["filter_stable"]
    assert math.isfinite(e["e_log_abs_A"])
