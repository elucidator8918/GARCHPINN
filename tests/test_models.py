"""Parameter recovery, forecasting and filter-consistency tests (float64)."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float64)

from garchpinn.data import Dataset  # noqa: E402
from garchpinn.losses import moment_residuals  # noqa: E402
from garchpinn.models import EGARCHPINN, GARCHPINN  # noqa: E402
from garchpinn.train import stage1_qmle  # noqa: E402

# Typical estimates: HHS (2012) log-linear Realized GARCH, HH (2016) Realized EGARCH.
TRUE = {
    "garch": dict(mu=0.03, omega=0.07, beta=0.55, gamma=0.41, xi=-0.18, phi=1.04,
                  tau1=-0.07, tau2=0.07, sigma_u=0.38),
    "egarch": dict(mu=0.03, omega=0.0, beta=0.97, gamma=0.30, tau1=-0.10, tau2=0.04,
                   xi=-0.40, phi=1.0, delta1=-0.07, delta2=0.07, sigma_u=0.38),
}
CLS = {"garch": GARCHPINN, "egarch": EGARCHPINN}
# absolute tolerances at T=5000 (several asymptotic s.e.)
TOL = {"mu": 0.05, "persistence": 0.015, "gamma": 0.08, "phi": 0.15, "xi": 0.25,
       "sigma_u": 0.02, "tau1": 0.03, "tau2": 0.03, "delta1": 0.03, "delta2": 0.03}
T_SIM = 5000


def true_model(v, **kw):
    return CLS[v](use_nn=False, **kw).load_params(**TRUE[v])


@pytest.fixture(scope="module", params=["garch", "egarch"])
def sim(request):
    v = request.param
    m = true_model(v)
    r, lx, lh = m.simulate(T_SIM, seed=1)
    return v, m, r, lx, lh


def test_parameter_recovery(sim):
    v, truth, r, lx, lh = sim
    ds = Dataset(r, lx, r.new_zeros(T_SIM, 0), None, n_train=T_SIM, n_val=0)
    cfg = SimpleNamespace(stage1_streams=1, stage1_burn_in=0, stage1_iters=300)
    m = CLS[v](use_nn=False)  # starts from the default init, not the truth
    stage1_qmle(m, ds, cfg)
    est, tru = m.params(), truth.params()
    for k, tol in TOL.items():
        if k in est:
            assert abs(est[k] - tru[k]) < tol, (v, k, est[k], tru[k])
    with torch.no_grad():
        out, _ = m(r, lx)
    corr = np.corrcoef(out["log_h"][100:].numpy(), lh[100:].numpy())[0, 1]
    assert corr > 0.95, corr


@pytest.mark.parametrize("v", ["garch", "egarch"])
def test_forecast_matches_monte_carlo(v):
    m = true_model(v)
    r, lx, _ = m.simulate(500, seed=2)
    with torch.no_grad():
        out, state = m(r, lx)
    H = 22
    fc = m.forecast(out["log_h_next"], H)
    mc = m.simulate_forecast(state, H, n_paths=400_000, generator=torch.Generator().manual_seed(3))
    for k in (1, 5, 22):
        for key in ("h", "x"):
            a, b = float(fc[key][k - 1]), float(mc[key][k - 1])
            assert abs(a / b - 1) < 0.015, (v, key, k, a, b)


@pytest.mark.parametrize("v", ["garch", "egarch"])
def test_forecast_no_tau2_drift(v):
    """Starting at the stationary mean of log h, the conditional mean of log h stays there:
    E(z^2 - 1) = 0, so tau2 / delta2 must not shift m_k."""
    m = true_model(v)
    c, a, *_ = m.state_space()
    mbar = (c / (1 - a)).detach()
    fc = m.forecast(mbar.reshape(1), 200)
    assert torch.allclose(fc["log_h"], mbar.expand(1, 200), atol=1e-10)
    # and the MC mean of log h agrees (drift would show as a level shift)
    m.load_params(**{**TRUE[v], "tau2": 0.3})
    c, a, *_ = m.state_space()
    mbar = (c / (1 - a)).detach()
    fc = m.forecast(mbar.reshape(1), 50)
    assert torch.allclose(fc["log_h"], mbar.expand(1, 50), atol=1e-10)


def test_moment_residuals_near_zero(sim):
    v, truth, r, lx, _ = sim
    with torch.no_grad():
        out, _ = truth(r, lx)
        mask = torch.ones(T_SIM)
        mask[:50] = 0
        rm = float(moment_residuals(out["z"], out["u"], truth.sigma_u, mask, lags=5))
        bad = CLS[v](use_nn=False).load_params(**{**TRUE[v], "sigma_u": 0.38, "gamma": 0.05})
        out_b, _ = bad(r, lx)
        rm_bad = float(moment_residuals(out_b["z"], out_b["u"], bad.sigma_u, mask, lags=5))
    # 20 conditions, each a mean of O(1)-variance terms over 5000 obs => E R ~ 20 * 2 / 5000
    assert rm < 0.03, rm
    assert rm_bad > rm


@pytest.mark.parametrize("v", ["garch", "egarch"])
def test_carried_state_equals_full_sequence(v):
    torch.manual_seed(0)
    m = CLS[v](n_exo=3, use_nn=True).load_params(**TRUE[v])
    torch.nn.init.normal_(m.net[-1].weight, std=0.5)  # make the correction non-trivial
    r, lx, _ = true_model(v).simulate(300, seed=4)
    exo = torch.randn(300, 3)
    with torch.no_grad():
        full, st_full = m(r, lx, exo)
        assert full["g"].abs().max() > 1e-3
        # same initial log h for both runs: init_log_h uses the first 22 obs of the first piece
        o1, st = m(r[:120], lx[:120], exo[:120])
        o2, st2 = m(r[120:], lx[120:], exo[120:], st)
        # batched (B, L) path too
        o2b, _ = m(r[None, 120:], lx[None, 120:], exo[None, 120:], {k: x[None] for k, x in st.items()})
    for k in ("log_h", "z", "u", "g"):
        cat = torch.cat([o1[k], o2[k]])
        assert torch.allclose(cat, full[k], atol=1e-12), k
        assert torch.allclose(o2b[k][0], o2[k], atol=1e-12), k
    assert torch.allclose(o2["log_h_next"], full["log_h_next"], atol=1e-12)
    for k in st_full:
        assert torch.allclose(st2[k], st_full[k], atol=1e-12), k
