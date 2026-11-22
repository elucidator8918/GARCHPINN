"""Tests for garchpinn/diagnostics.py (numpy stats, s.e., backtests)."""
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

torch.set_default_dtype(torch.float64)

from garchpinn import diagnostics as D  # noqa: E402
from garchpinn.data import Dataset  # noqa: E402
from garchpinn.models import GARCHPINN  # noqa: E402
from garchpinn.train import stage1_qmle  # noqa: E402


def test_distributions():
    assert abs(D.norm_ppf(0.975) - 1.959963984540054) < 1e-9
    assert abs(D.norm_ppf(0.01) + 2.326347874040841) < 1e-9
    assert abs(D.norm_cdf(1.959963984540054) - 0.975) < 1e-12
    assert abs(D.chi2_sf(3.841458820694124, 1) - 0.05) < 1e-9
    assert abs(D.chi2_sf(5.991464547107979, 2) - 0.05) < 1e-9
    assert abs(D.chi2_sf(33.92443847, 22) - 0.05) < 1e-6
    assert abs(D.chi2_ppf(0.95, 1) - 3.8414588) < 1e-5


def test_ljung_box_iid_does_not_reject():
    rng = np.random.default_rng(1)
    lb = D.ljung_box(rng.standard_normal(3000), (10, 22))
    assert lb[10]["pvalue"] > 0.05 and lb[22]["pvalue"] > 0.05


def test_ljung_box_rejects_ar1():
    rng = np.random.default_rng(2)
    e = rng.standard_normal(2000)
    x = np.zeros_like(e)
    for t in range(1, len(e)):
        x[t] = 0.3 * x[t - 1] + e[t]
    assert D.ljung_box(x, (10,))[10]["pvalue"] < 1e-6


def test_kupiec_exact_rate_does_not_reject():
    hits = np.zeros(1000, bool)
    hits[::20] = True  # exactly 5%
    k = D.kupiec(hits, 0.05)
    assert k["LR_uc"] < 1e-9 and k["pvalue"] > 0.99
    assert D.kupiec(hits, 0.01)["pvalue"] < 1e-6


def test_christoffersen_detects_clustering():
    hits = np.zeros(1000, int)
    hits[100:150] = 1  # 5% in one cluster
    assert D.christoffersen(hits, 0.05)["pvalue_ind"] < 1e-6


def test_mincer_zarnowitz_unbiased():
    rng = np.random.default_rng(3)
    f = np.exp(rng.standard_normal(3000))
    y = f * np.exp(rng.standard_normal(3000) * 0.3 - 0.045)
    r = D.mincer_zarnowitz(y, f)
    assert abs(r["b"] - 1) < 0.1 and r["wald_pvalue"] > 0.01


def test_mcs_keeps_best_drops_bad():
    rng = np.random.default_rng(4)
    n = 1000
    base = rng.standard_normal(n)
    L = {"good": base, "same": base + 0.01 * rng.standard_normal(n), "bad": base + 1.0}
    m = D.model_confidence_set(L, B=200)
    assert "good" in m["mcs"] and "bad" not in m["mcs"]


@pytest.fixture(scope="module")
def fitted():
    gen = GARCHPINN(use_nn=False).load_params(mu=0.03, omega=0.07, beta=0.55, gamma=0.41, xi=-0.18,
                                              phi=1.04, tau1=-0.07, tau2=0.07, sigma_u=0.38)
    T = 2500
    r, lx, _ = gen.simulate(T, seed=7)
    ds = Dataset(r, lx, torch.zeros(T, 0), list(range(T)), T, 0)
    m = GARCHPINN(use_nn=False)
    stage1_qmle(m, ds, SimpleNamespace(stage1_streams=1, stage1_iters=300, stage1_burn_in=22))
    return m, ds


def test_sandwich_close_to_hessian(fitted):
    m, ds = fitted
    se = D.standard_errors(m, ds)
    ratio = np.array(se["ratio"])
    assert np.all(np.isfinite(ratio))
    assert np.all((ratio > 0.75) & (ratio < 1.33)), dict(zip(se["names"], ratio))
    assert se["mean_score_norm"] < 1e-2


def test_profile_minimum_near_estimate(fitted):
    m, ds = fitted
    p = D.profile_likelihood(m, ds, params=("gamma",), n_grid=5, iters=5, max_obs=1000)
    lr = np.array(p["profiles"]["gamma"]["lr"])
    assert np.argmin(lr) == 2 and lr[0] > 1 and lr[-1] > 1


def test_impulse_response_matches_mc_for_pure_model(fitted):
    m, ds = fitted
    with torch.no_grad():
        _, st = m(ds.r, ds.log_x, ds.exo)
    irf = D.impulse_response(m, 10, state=st, n_paths=500, eps=0.1)
    assert np.allclose(irf["du_mc"], irf["du"], atol=0.02)
    assert abs(irf["half_life"] - math.log(0.5) / math.log(float(m.persistence))) < 1e-9


def test_moment_paths_all_lags():
    rng = np.random.default_rng(5)
    z, u = rng.standard_normal(200), rng.standard_normal(200)
    m = D.moment_paths(z, u, lags=5)
    assert len(m) == 5 + 3 * 5
    for l in range(1, 6):
        s = m[f"u_t z_t-{l}"]
        assert np.all(np.isnan(s[:l])) and np.allclose(s[l:], u[l:] * z[:-l])


def test_impulse_response_analytic_only_and_half_life(fitted):
    m, _ = fitted
    irf = D.impulse_response(m, 50, state=None)
    a = irf["pi"]
    assert np.allclose(irf["du"][1:] / irf["du"][:-1], a)
    kh = irf["k_half"]
    assert abs(irf["du"][0] * a ** (kh - 1) - irf["du"][0] / 2) < 1e-12
    assert "du_mc" not in irf


def test_recovery_zscores():
    rec = {"true": {"a": 1.0, "b": 0.0},
           "T": {100: [{"a": 1.2, "b": -0.1}, {"a": 0.9, "b": 0.3}]},
           "se": {100: [{"a": 0.1, "b": 0.1}, {"a": 0.1, "b": float("nan")}]}}
    z = D.recovery_zscores(rec)
    assert np.allclose(z["a"], [2.0, -1.0]) and np.allclose(z["b"], [-1.0])


def test_plots_render(tmp_path, fitted):
    from garchpinn import plots as P
    m, ds = fitted
    se = D.standard_errors(m, ds)
    st = D.stationarity_point(m, se)
    st_big = dict(st, cov=(np.array(st["cov"]) * 400).tolist())  # ellipse far wider than the old pad
    for s in (st, st_big):
        P.plot_I5(s, str(tmp_path))
    rec = {"true": {"beta": 0.5}, "T": {100: [{"beta": 0.55}, {"beta": 0.4}]},
           "se": {100: [{"beta": 0.05}, {"beta": 0.05}]}}
    P.plot_I3(rec, str(tmp_path))
    P.plot_F3(D.impulse_response(m, 30), str(tmp_path))
    assert (tmp_path / "I3_parameter_recovery.png").exists() and (tmp_path / "F3_impulse_response.png").exists()


def test_i5_limits_contain_ellipse(tmp_path, fitted):
    import matplotlib.pyplot as plt
    from garchpinn import plots as P
    m, ds = fitted
    se = D.standard_errors(m, ds)
    for variant in ("garch", "egarch"):
        st = dict(D.stationarity_point(m, se), variant=variant)
        st["cov"] = (np.array(st["cov"]) * 400).tolist()
        captured = {}
        orig = P._save
        P._save = lambda fig, od, fn: captured.setdefault("ax", fig.axes[0])
        try:
            P.plot_I5(st, str(tmp_path))
        finally:
            P._save = orig
        a = captured["ax"]
        line = [l for l in a.get_lines() if l.get_label() == "95% sandwich ellipse"][0]
        xd, yd = line.get_xdata(), line.get_ydata()
        (x0, x1), (y0, y1) = a.get_xlim(), a.get_ylim()
        assert x0 <= xd.min() and xd.max() <= x1 and y0 <= yd.min() and yd.max() <= y1
        plt.close("all")


def test_forecast_mgf_defaults():
    from garchpinn.config import Config
    assert Config(freq="daily").forecast_mgf == "gaussian"
    assert Config(freq="5min").forecast_mgf == "empirical"


def test_evaluate_reports_other_mgf(fitted):
    from garchpinn.evaluate import evaluate
    m, ds0 = fitted
    T = ds0.T
    ds = Dataset(ds0.r, ds0.log_x, ds0.exo, list(range(T)), 2000, 0)
    with torch.no_grad():
        out, _ = m(ds.r, ds.log_x, ds.exo)
    m.set_empirical_mgf(out["z"][:2000], out["u"][:2000])
    for mgf, alt in (("gaussian", "model_emp"), ("empirical", "model_gauss")):
        cfg = SimpleNamespace(horizons=(1, 5), har_windows=(1, 5, 22), stage1_iters=50, forecast_mgf=mgf)
        r = evaluate(m, ds, cfg)
        assert r["forecast_mgf"] == mgf and alt in r["k=5"] and "model" in r["k=5"]


def test_ablation_variants_valid_config():
    from garchpinn.ablation import VARIANTS
    from garchpinn.config import Config
    for name, (_, over, _) in VARIANTS.items():
        for k in over:
            assert hasattr(Config(), k), (name, k)
        Config(**over)
    assert {"overnight_drop", "moment_raw", "no_warmup"} <= set(VARIANTS)
