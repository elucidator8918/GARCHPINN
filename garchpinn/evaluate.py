"""Out-of-sample evaluation (reference doc Section 8).

Losses (Patton 2011): QLIKE = log f + x/f, normalised QLIKE = x/f - log(x/f) - 1, MSE.
Targets: sum_{i=1..k} x_{t+i}; forecasts: sum_{i=1..k} E_t x_{t+i} (or E_t h for GARCH(1,1)).
The model filter runs once over the whole series with fixed parameters, so the state at the
start of the test period carries the full-sample history (no re-initialisation).
"""
import math

import numpy as np
import torch

from .benchmarks import GARCH11, LogHAR


def qlike_norm(x, f):
    q = x / f
    return q - torch.log(q) - 1 if torch.is_tensor(q) else q - np.log(q) - 1


def qlike(x, f):
    return (torch.log(f) if torch.is_tensor(f) else np.log(f)) + x / f


def mse(x, f):
    return (x - f) ** 2


def _np(a):
    return a.detach().cpu().numpy().astype(float) if torch.is_tensor(a) else np.asarray(a, float)


def _nw_bandwidth(d):
    """Newey-West (1994) automatic Bartlett bandwidth, n_pre = 4 (n/100)^{2/9}."""
    n = len(d)
    dc = d - d.mean()
    npre = int(math.floor(4 * (n / 100) ** (2 / 9)))
    g = np.array([dc[j:] @ dc[: n - j] / n for j in range(npre + 1)])
    s0 = g[0] + 2 * g[1:].sum()
    s1 = 2 * (np.arange(1, npre + 1) * g[1:]).sum()
    if s0 <= 0:
        return npre
    return int(math.ceil(1.1447 * ((s1 / s0) ** 2 * n) ** (1 / 3)))


def _betacf(a, b, x):
    """Continued fraction for the incomplete beta (Numerical Recipes 6.4)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 500):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / (c if abs(c) > tiny else tiny)
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > tiny else tiny)
        c = 1 + aa / (c if abs(c) > tiny else tiny)
        de = d * c
        h *= de
        if abs(de - 1) < 1e-14:
            break
    return h


def _betainc(a, b, x):
    if x <= 0 or x >= 1:
        return float(x >= 1)
    lb = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x)
    if x < (a + 1) / (a + b + 2):
        return math.exp(lb) * _betacf(a, b, x) / a
    return 1 - math.exp(lb) * _betacf(b, a, 1 - x) / b


def _t_sf2(t, df):
    """Two-sided Student-t p-value P(|T_df| > t) = I_{df/(df+t^2)}(df/2, 1/2) (no scipy)."""
    return float(min(1.0, max(0.0, _betainc(df / 2, 0.5, df / (df + t * t)))))


def dm_test(loss_a, loss_b, k=1, lag=None):
    """Diebold-Mariano test of E[loss_a - loss_b] = 0 for k-step (overlapping) forecasts.

    Long-run variance: Bartlett kernel with bandwidth max(NW94 automatic, 2(k-1), 1)
    (a Bartlett kernel truncated at k-1 underestimates the MA(k-1) long-run variance),
    Harvey-Leybourne-Newbold (1997) small-sample factor and Student-t(n-1) p-values.
    Negative stat => model a has lower loss. Returns stat, pvalue, lag used."""
    d = _np(loss_a) - _np(loss_b)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 3:
        return {"stat": float("nan"), "pvalue": float("nan"), "lag": 0}
    if lag is None:
        lag = max(_nw_bandwidth(d), 2 * (k - 1), 1)
    lag = min(lag, n - 1)
    dc = d - d.mean()
    v = dc @ dc / n
    for j in range(1, lag + 1):
        v += 2 * (1 - j / (lag + 1)) * (dc[j:] @ dc[:-j]) / n
    if v <= 0:
        return {"stat": float("nan"), "pvalue": float("nan"), "lag": int(lag)}
    stat = d.mean() / math.sqrt(v / n)
    hln = math.sqrt(max((n + 1 - 2 * k + k * (k - 1) / n) / n, 1e-12))
    stat *= hln
    return {"stat": float(stat), "pvalue": float(_t_sf2(abs(stat), n - 1)), "lag": int(lag)}


@torch.no_grad()
def _model_sum_forecasts(model, ds, horizons, mgf="gaussian"):
    """Filter the full series once; return out, and for each k the (T,) array of
    sum_{i<=k} E_t x_{t+i} indexed by origin t (NaN where log_h[t+1] is unavailable).
    log h_{t+1} already contains the known g_t; steps >= 2 use the training mean g_bar
    (approximate for the PINN, exact for the pure model)."""
    out, _ = model(ds.r, ds.log_x, ds.exo)
    T = ds.T
    lh1 = out["log_h"][1:]  # log h_{t+1}, origins t = 0..T-2
    K = max(horizons)
    gb = model.g_bar if model.nn_active else 0.0
    fc = model.forecast(lh1, K, g_bar=gb, mgf=mgf)["x"]  # (T-1, K)
    cs = torch.cumsum(fc, -1)
    res, series = {}, {}
    for k in horizons:
        a = np.full(T, np.nan)
        a[: T - 1] = _np(cs[:, k - 1])
        res[k] = a
    return out, res


def _scores(x, f):
    return {"qlike_norm": qlike_norm(x, f), "qlike": qlike(x, f), "mse": mse(x, f)}


def evaluate(model, ds, cfg, pure_model=None, garch=None, har=None, return_series=False):
    """Returns nested dict:
      {"k=<k>": {"n": int, "<name>": {qlike_norm, qlike, mse},
                 "dm": {"model_vs_<bench>": {stat, pvalue}}},
       "loglik": {"<name>": {"l_r": per-obs, "l_rx": per-obs}},
       "garch11": params}
    Benchmarks are fitted on the 'train' split (same estimation window end as the Realized
    models) unless passed in pre-fitted. GARCH(1,1) forecasts E h; it is scored after the
    proxy map c_hat = mean(x)/mean(h) estimated on the train split ("garch11"), and also
    unscaled ("garch11_raw"). The primary "<name>" forecast uses cfg.forecast_mgf (default
    "gaussian", "empirical" for 5-min); the other MGF is reported as "<name>_emp" (primary
    Gaussian) or "<name>_gauss" (primary empirical). Without stored residuals the primary
    falls back to Gaussian.
    return_series=True returns (res, series) with series["k=<k>"] = {"t": origins, "target",
    "forecast": {name: (n,)}, "loss": {name: {qlike, mse, ...}: (n,)}} for per-t diagnostics."""
    model.eval()
    fit = ds.split("train")
    te = ds.split("test")
    T, t0 = ds.T, te.start
    horizons = tuple(cfg.horizons)
    x = np.exp(_np(ds.log_x))
    csx = np.concatenate([[0.0], np.cumsum(x)])

    if garch is None:
        garch = GARCH11().fit(ds.r[fit], iters=getattr(cfg, "stage1_iters", 200))
    h_g = garch.filter(ds.r)  # h_g[t+1] = forecast for t+1 made at t
    if har is None:
        har = LogHAR(cfg.har_windows).fit(_np(ds.log_x), fit.stop, horizons)

    models = {"model": model}
    if pure_model is not None:
        models["pure"] = pure_model
    fcs, outs = {}, {}
    primary = getattr(cfg, "forecast_mgf", "gaussian")
    for name, m in models.items():
        has_emp = m.emp_resid is not None
        mg = primary if has_emp else "gaussian"
        outs[name], fcs[name] = _model_sum_forecasts(m, ds, horizons, mgf=mg)
        if has_emp:
            alt, suffix = ("empirical", "_emp") if mg == "gaussian" else ("gaussian", "_gauss")
            _, fcs[name + suffix] = _model_sum_forecasts(m, ds, horizons, mgf=alt)
    with torch.no_grad():
        c_hat = float(np.mean(x[fit]) / _np(h_g[: fit.stop]).mean())

    res, series = {}, {}
    for k in horizons:
        # origins t with t+1 >= t0 (forecast made with info up to t, target inside test) and t+k <= T-1
        t = np.arange(t0 - 1, T - k)
        tgt = csx[t + k + 1] - csx[t + 1]
        cand = {n: fcs[n][k][t] for n in fcs}
        gf = _np(garch.forecast(h_g[torch.as_tensor(t + 1, device=h_g.device)], k).sum(-1))
        cand["garch11"] = c_hat * gf
        cand["garch11_raw"] = gf
        cand["har"] = har.predict_sum(_np(ds.log_x), k)[t]
        losses = {n: _scores(tgt, f) for n, f in cand.items()}
        rk = {"n": int(len(t))}
        for n, L in losses.items():
            rk[n] = {m: float(np.nanmean(v)) for m, v in L.items()}
        rk["dm"] = {f"model_vs_{b}": dm_test(losses["model"]["qlike"], losses[b]["qlike"], k=k)
                    for b in losses if b != "model"}
        res[f"k={k}"] = rk
        series[f"k={k}"] = {"t": t, "target": tgt, "forecast": cand, "loss": losses}

    mask = torch.zeros(T, dtype=ds.r.dtype, device=ds.r.device)
    mask[te] = 1.0
    ll = {}
    with torch.no_grad():
        for name, m in models.items():
            lr, lx = m.nll_parts(outs[name], mask)
            ll[name] = {"l_r": float(-lr), "l_rx": float(-(lr + lx))}
        eps2 = (ds.r - garch.mu) ** 2
        hg = h_g[:-1]
        lrg = -0.5 * (math.log(2 * math.pi) + torch.log(hg) + eps2 / hg)
        ll["garch11"] = {"l_r": float(lrg[te].mean())}
    res["loglik"] = ll
    res["garch11"] = {**garch.params(), "proxy_scale_c_hat": c_hat}
    res["forecast_mgf"] = primary
    return (res, series) if return_series else res
