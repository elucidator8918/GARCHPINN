"""Numeric diagnostics behind the figures of doc Section "Expected figures and diagnostics".

Everything is numpy / torch only (no scipy): normal CDF / quantile, chi-square survival
function, Ljung-Box, Jarque-Bera, robust (sandwich) vs Hessian standard errors, profile
likelihood, VaR backtests (Kupiec, Christoffersen), Mincer-Zarnowitz, an approximate Model
Confidence Set, calibration by deciles, cumulative DM loss differentials, impulse responses,
parameter recovery (I3) and rolling-window parameters (I4).
"""
import copy
import math
from types import SimpleNamespace

import numpy as np
import torch

from .evaluate import _nw_bandwidth
from .models.base import LOG2PI, PERSIST_MAX


# ======================================================================== distributions
def norm_cdf(x):
    x = np.asarray(x, float)
    return 0.5 * np.vectorize(math.erfc)(-x / math.sqrt(2.0))


def norm_pdf(x):
    x = np.asarray(x, float)
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def norm_ppf(p):
    """Acklam's rational approximation plus one Newton step (|error| < 1e-12)."""
    p = np.asarray(p, float)
    a = [-3.969683028665376e01, 2.209460984245205e02, -2.759285104469687e02, 1.383577518672690e02,
         -3.066479806614716e01, 2.506628277459239e00]
    b = [-5.447609879822406e01, 1.615858368580409e02, -1.556989798598866e02, 6.680131188771972e01,
         -1.328068155288572e01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e00, -2.549732539343734e00,
         4.374664141464968e00, 2.938163982698783e00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e00, 3.754408661907416e00]
    pl = 0.02425
    x = np.empty_like(p)
    lo, hi = p < pl, p > 1 - pl
    mid = ~(lo | hi)
    q = np.sqrt(-2 * np.log(np.where(lo, p, 0.5)))
    x = np.where(lo, (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                 / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1), x)
    q = np.sqrt(-2 * np.log(np.where(hi, 1 - p, 0.5)))
    x = np.where(hi, -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
                 / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1), x)
    q = p - 0.5
    rr = q * q
    x = np.where(mid, (((((a[0] * rr + a[1]) * rr + a[2]) * rr + a[3]) * rr + a[4]) * rr + a[5]) * q
                 / (((((b[0] * rr + b[1]) * rr + b[2]) * rr + b[3]) * rr + b[4]) * rr + 1), x)
    e = norm_cdf(x) - p  # Newton refinement
    x = x - e / np.maximum(norm_pdf(x), 1e-300)
    return np.where(p <= 0, -np.inf, np.where(p >= 1, np.inf, x))


def _gammainc_upper(a, x):
    """Regularized upper incomplete gamma Q(a, x) (Numerical Recipes 6.2)."""
    if x <= 0:
        return 1.0
    gln = math.lgamma(a)
    if x < a + 1:  # series for P
        ap, s, de = a, 1.0 / a, 1.0 / a
        for _ in range(1000):
            ap += 1
            de *= x / ap
            s += de
            if abs(de) < abs(s) * 1e-15:
                break
        return max(0.0, 1.0 - s * math.exp(-x + a * math.log(x) - gln))
    tiny = 1e-300  # continued fraction for Q
    b = x + 1 - a
    c, d = 1 / tiny, 1 / b
    h = d
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2
        d = an * d + b
        d = tiny if abs(d) < tiny else d
        c = b + an / c
        c = tiny if abs(c) < tiny else c
        d = 1 / d
        de = d * c
        h *= de
        if abs(de - 1) < 1e-15:
            break
    return min(1.0, math.exp(-x + a * math.log(x) - gln) * h)


def chi2_sf(x, df):
    return float(_gammainc_upper(df / 2.0, max(float(x), 0.0) / 2.0)) if np.isfinite(x) else float("nan")


def chi2_ppf(p, df):
    lo, hi = 0.0, 1000.0
    for _ in range(200):
        m = 0.5 * (lo + hi)
        lo, hi = (m, hi) if 1 - chi2_sf(m, df) < p else (lo, m)
    return 0.5 * (lo + hi)


# ======================================================================== time-series stats
def acf(x, nlags):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    xc = x - x.mean()
    m = 1 << int(np.ceil(np.log2(2 * n)))
    f = np.fft.rfft(xc, m)
    c = np.fft.irfft(f * np.conj(f), m)[: nlags + 1]
    return c / c[0] if c[0] > 0 else np.zeros(nlags + 1)


def ljung_box(x, lags=(10, 22)):
    """Ljung-Box Q(m) = n(n+2) sum_{j<=m} rho_j^2/(n-j) ~ chi2(m). Returns {m: {Q, pvalue}}."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    lags = [int(m) for m in np.atleast_1d(lags) if m < n]
    if not lags:
        return {}
    rho = acf(x, max(lags))
    out = {}
    for m in lags:
        j = np.arange(1, m + 1)
        q = float(n * (n + 2) * np.sum(rho[1: m + 1] ** 2 / (n - j)))
        out[m] = {"Q": q, "pvalue": chi2_sf(q, m)}
    return out


def jarque_bera(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    xc = x - x.mean()
    s2 = np.mean(xc ** 2)
    sk = np.mean(xc ** 3) / s2 ** 1.5
    ku = np.mean(xc ** 4) / s2 ** 2
    jb = n / 6 * (sk ** 2 + (ku - 3) ** 2 / 4)
    return {"JB": float(jb), "pvalue": chi2_sf(jb, 2), "skew": float(sk), "kurt": float(ku)}


def hac_lrv(d, lag=None):
    """Bartlett (Newey-West) long-run variance of d."""
    d = np.asarray(d, float)
    n = len(d)
    if lag is None:
        lag = _nw_bandwidth(d)
    lag = int(min(max(lag, 0), n - 1))
    dc = d - d.mean()
    v = dc @ dc / n
    for j in range(1, lag + 1):
        v += 2 * (1 - j / (lag + 1)) * (dc[j:] @ dc[:-j]) / n
    return float(v)


def _hac_cov(X, e, lag):
    """Newey-West covariance of OLS coefficients."""
    n = len(e)
    S = X * e[:, None]
    Om = S.T @ S / n
    for j in range(1, lag + 1):
        G = S[j:].T @ S[:-j] / n
        Om += (1 - j / (lag + 1)) * (G + G.T)
    XtX = np.linalg.inv(X.T @ X / n)
    return XtX @ Om @ XtX / n


def cross_corr(a, b, max_lag=10):
    """corr(a_t, b_{t+j}) for j in [-max_lag, max_lag]."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a = (a - a.mean()) / a.std()
    b = (b - b.mean()) / b.std()
    n = len(a)
    js = np.arange(-max_lag, max_lag + 1)
    cc = [np.mean(a[: n - j] * b[j:]) if j >= 0 else np.mean(a[-j:] * b[: n + j]) for j in js]
    return js, np.array(cc)


def residuals(out, model):
    """z_t and u_t / sigma_u as numpy arrays."""
    z = out["z"].detach().cpu().numpy().ravel()
    u = (out["u"] / model.sigma_u).detach().cpu().numpy().ravel()
    return z, u


def residual_stats(z, u, lb_lags=(10, 22)):
    """R1/R2/R5 numbers: JB of z and u~, Ljung-Box of z, z^2, u~, u~^2, z u~, corr(z, u~)."""
    series = {"z": z, "z2": z ** 2, "u": u, "u2": u ** 2, "zu": z * u}
    return {"jb": {"z": jarque_bera(z), "u": jarque_bera(u)},
            "ljung_box": {k: {str(m): v for m, v in ljung_box(s, lb_lags).items()} for k, s in series.items()},
            "corr_z_u": float(np.corrcoef(z, u)[0, 1])}


def moment_paths(z, u, lags=1):
    """Per-t moment conditions of eq. (moments) (as in losses.moment_residuals, u standardised):
    returns {name: (T,) series m_{j,t}} (lagged products start at t = l)."""
    e = z * z - 1
    m = {"z": z, "z^2-1": e, "u": u, "u z": u * z, "u (z^2-1)": u * e}
    for l in range(1, lags + 1):
        pad = np.full(l, np.nan)
        m[f"e_t e_t-{l}"] = np.concatenate([pad, e[l:] * e[:-l]])
        m[f"u_t u_t-{l}"] = np.concatenate([pad, u[l:] * u[:-l]])
        m[f"u_t z_t-{l}"] = np.concatenate([pad, u[l:] * z[:-l]])
    return m


def running_mean_band(m):
    """Running mean of m and its +-2 HAC s.e. (full-sample long-run variance / t)."""
    ok = np.isfinite(m)
    x = np.where(ok, m, 0.0)
    cnt = np.cumsum(ok)
    rm = np.cumsum(x) / np.maximum(cnt, 1)
    lrv = hac_lrv(m[ok])
    return rm, 2 * np.sqrt(lrv / np.maximum(cnt, 1))


# ======================================================================== econ parameter vector
def _display_name(model, n):
    if n == "persist_raw":
        return "persistence"
    if n == "log_sigma_u":
        return "sigma_u"
    if model.variant == "garch":
        return {"d1": "tau1", "d2": "tau2"}.get(n, n)
    return {"d1": "delta1", "d2": "delta2", "t1": "tau1", "t2": "tau2"}.get(n, n)


def econ_vector(model):
    """Natural-scale econ parameters (sigma_u and persistence untransformed): names, raw names, theta."""
    raw = [n for n, _ in model.named_parameters() if not n.startswith("net.")]
    vals = []
    for n in raw:
        p = dict(model.named_parameters())[n].detach()
        vals.append(model.sigma_u.detach() if n == "log_sigma_u" else
                    model.persistence.detach() if n == "persist_raw" else p)
    return [_display_name(model, n) for n in raw], raw, torch.stack([v.reshape(()) for v in vals])


def _raw_dict(raw, theta):
    d = {}
    for i, n in enumerate(raw):
        v = theta[i]
        if n == "log_sigma_u":
            v = torch.log(v.clamp_min(1e-8))
        elif n == "persist_raw":
            q = (v / PERSIST_MAX).clamp(1e-9, 1 - 1e-9)
            v = torch.log(q) - torch.log1p(-q)
        d[n] = v
    return d


def _ll_fn(model, raw, r, lx, ex, burn_in):
    """theta -> per-observation joint log-likelihood (n - burn_in,) of the pure model."""
    sidx = raw.index("log_sigma_u")

    def f(theta):
        out, _ = torch.func.functional_call(model, _raw_dict(raw, theta), (r, lx, ex), strict=False)
        su2 = theta[sidx].clamp_min(1e-8) ** 2
        lr = 0.5 * (LOG2PI + out["log_h"] + out["z"] ** 2)
        lxx = 0.5 * (LOG2PI + torch.log(su2) + out["u"] ** 2 / su2)
        return -(lr + lxx)[burn_in:]
    return f


def _train_window(ds, max_obs):
    n = min(ds.n_train, int(max_obs))
    return ds.r[:n], ds.log_x[:n], ds.exo[:n]


def standard_errors(model, ds, burn_in=22, max_obs=6000):
    """Hessian vs sandwich (robust) s.e. of the stage-1 econ parameters (doc I2).

    H = d^2 sum_t l_t / d theta^2 via torch.autograd.functional.hessian; J = sum_t s_t s_t' from
    per-observation scores (forward-mode jacobian). cov_H = (-H)^-1, cov_S = H^-1 J H^-1.
    Uses the first min(n_train, max_obs) training observations, NN off."""
    m = copy.deepcopy(model)
    m.nn_active = False
    names, raw, theta = econ_vector(m)
    r, lx, ex = _train_window(ds, max_obs)
    f = _ll_fn(m, raw, r, lx, ex, burn_in)
    H = torch.autograd.functional.hessian(lambda th: f(th).sum(), theta)
    try:
        S = torch.autograd.functional.jacobian(f, theta, vectorize=True, strategy="forward-mode")
    except Exception:  # fallback: one forward-mode jvp per parameter
        S = torch.stack([torch.func.jvp(f, (theta,), (torch.eye(len(theta), dtype=theta.dtype, device=theta.device)[i],))[1]
                         for i in range(len(theta))], 1)
    Hn, Sn = H.detach().cpu().numpy(), S.detach().cpu().numpy()
    Ainv = np.linalg.pinv(-Hn)
    J = Sn.T @ Sn
    cov_h, cov_s = Ainv, Ainv @ J @ Ainv
    se_h, se_s = np.sqrt(np.abs(np.diag(cov_h))), np.sqrt(np.abs(np.diag(cov_s)))
    return {"names": names, "estimate": theta.detach().cpu().numpy().tolist(), "se_hessian": se_h.tolist(),
            "se_sandwich": se_s.tolist(), "ratio": (se_s / np.maximum(se_h, 1e-300)).tolist(),
            "cov_hessian": cov_h.tolist(), "cov_sandwich": cov_s.tolist(), "n": int(Sn.shape[0]),
            "mean_score_norm": float(np.linalg.norm(Sn.mean(0)))}


def _lbfgs(f, free0, iters):
    x = free0.clone().requires_grad_(True)
    opt = torch.optim.LBFGS([x], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe",
                            tolerance_grad=1e-9, tolerance_change=1e-12)

    def closure():
        opt.zero_grad()
        v = f(x)
        if not torch.isfinite(v):
            v = torch.tensor(1e10, dtype=x.dtype, device=x.device, requires_grad=True) + 0 * x.sum()
        v.backward()
        return v
    try:
        opt.step(closure)
    except Exception:
        pass
    with torch.no_grad():
        return x.detach(), float(f(x))


def profile_likelihood(model, ds, se=None, params=("persistence", "gamma", "phi", "sigma_u"),
                       n_grid=7, width=3.0, iters=10, burn_in=22, max_obs=2000):
    """Profile log-likelihood (doc I1): for each parameter on a grid of +-width s.e. (Hessian),
    re-optimise the others with `iters` LBFGS iterations. Returns {name: {grid, lr}} with
    lr = 2 (l_max - l_p), and the chi2_1 0.95 cut 3.84. Cheap by default (subset, few iters)."""
    m = copy.deepcopy(model)
    m.nn_active = False
    for p in m.parameters():
        p.requires_grad_(False)
    names, raw, theta = econ_vector(m)
    r, lx, ex = _train_window(ds, max_obs)
    f = _ll_fn(m, raw, r, lx, ex, burn_in)
    negll = lambda th: -f(th).sum()
    th_all, nll_max = _lbfgs(negll, theta, iters)  # optimum on this subset
    if se is None:
        se = standard_errors(model, ds, burn_in, max_obs)
    sed = dict(zip(se["names"], se["se_hessian"]))
    res = {}
    for name in params:
        if name not in names:
            continue
        i = names.index(name)
        s = max(sed.get(name, 0.01), 1e-4)
        grid = float(th_all[i]) + width * s * np.linspace(-1, 1, n_grid)
        if name == "persistence":
            grid = np.clip(grid, 1e-3, PERSIST_MAX - 1e-6)
        if name == "sigma_u":
            grid = np.clip(grid, 1e-3, None)
        free_idx = torch.tensor([j for j in range(len(theta)) if j != i], device=theta.device)
        prof = []
        for gv in grid:
            def fp(x, gv=gv):
                th = th_all.clone()
                th = th.index_put((free_idx,), x)
                th = torch.cat([th[:i], torch.tensor([gv], dtype=th.dtype, device=th.device), th[i + 1:]])
                return negll(th)
            _, v = _lbfgs(fp, th_all[free_idx], iters)
            prof.append(v)
        prof = np.array(prof)
        nll_max = min(nll_max, float(np.nanmin(prof)))
        res[name] = {"grid": grid.tolist(), "nll": prof.tolist(), "estimate": float(th_all[i])}
    for v in res.values():
        v["lr"] = (2 * (np.array(v["nll"]) - nll_max)).tolist()
    return {"profiles": res, "cut": chi2_ppf(0.95, 1), "n": int(len(r) - burn_in)}


def stationarity_point(model, se=None):
    """(x, y) coordinates of doc I5 and their sandwich covariance (delta method).
    RG: (beta, gamma phi) with beta = pi - gamma phi; REGARCH: (beta, gamma)."""
    p = model.params()
    if model.variant == "garch":
        x, y = p["beta"], p["gamma"] * p["phi"]
    else:
        x, y = p["beta"], p["gamma"]
    out = {"x": x, "y": y, "variant": model.variant}
    if se is not None:
        names = se["names"]
        C = np.array(se["cov_sandwich"])
        G = np.zeros((2, len(names)))
        ip, ig, ih = names.index("persistence"), names.index("gamma"), names.index("phi")
        if model.variant == "garch":
            G[0, ip], G[0, ig], G[0, ih] = 1.0, -p["phi"], -p["gamma"]
            G[1, ig], G[1, ih] = p["phi"], p["gamma"]
        else:
            G[0, ip], G[1, ig] = 1.0, 1.0
        out["cov"] = (G @ C @ G.T).tolist()
    return out


# ======================================================================== model-output helpers
@torch.no_grad()
def impulse_response(model, K, state=None, n_paths=2000, eps=0.1, seed=0):
    """d E_t log h_{t+k} / d u_t and / d z_t for k = 1..K (doc F3).
    Analytic (state-space, linear part): Bu a^{k-1} and A1 a^{k-1}; this is the primary IRF and is
    exact for the pure model. Half-life in k: k_half = 1 + log(1/2)/log(a). Optional overlay (if
    state given and n_paths > 0):
    Monte Carlo with common random numbers, shocking u_t (and log x_t by the same amount) by eps,
    with the NN correction active if the model has it on (inputs frozen at the last value)."""
    c, a, A1, A2, Bu = (float(v.detach()) for v in model.state_space())
    k = np.arange(1, K + 1)
    res = {"k": k, "du": Bu * a ** (k - 1), "dz": A1 * a ** (k - 1), "pi": a,
           "half_life": math.log(0.5) / math.log(a) if 0 < a < 1 else float("inf")}
    hl = res["half_life"]
    res["k_half"] = hl + 1  # du decays as a^(k-1): du(k_half) = du(1) / 2
    if state is not None and n_paths:
        base = _sim_logh(model, state, K, n_paths, seed, 0.0).mean(0)
        pert = _sim_logh(model, state, K, n_paths, seed, eps).mean(0)
        res["du_mc"] = (pert - base) / eps
    return res


@torch.no_grad()
def _sim_logh(model, state, K, n, seed, du0=0.0):
    """(n, K) simulated log h_{t+1..t+K} from a filter state; du0 shocks u_t and log x_t."""
    dev = state["lh"].device
    gen = torch.Generator(device=dev).manual_seed(seed)
    st = {k: (v.expand(n, *v.shape[-1:]) if k == "f" else v.expand(n)).clone() for k, v in state.items()}
    st["u"] = st["u"] + du0
    st["lx"] = st["lx"] + du0
    lh = model._bound(model.garch_step(st["lh"], st["lx"], st["z"], st["u"])
                      + model.correction(st["f"], st["z"], st["lh"]))
    out = []
    for _ in range(K):
        out.append(lh)
        z = torch.randn(n, generator=gen, dtype=lh.dtype, device=dev)
        u = torch.randn(n, generator=gen, dtype=lh.dtype, device=dev) * model.sigma_u
        lx = model.xi + model.phi * lh + model.meas_lev(z) + u
        lh = model._bound(model.garch_step(lh, lx, z, u) + model.correction(st["f"], z, lh))
    return torch.stack(out, 1).cpu().numpy()


@torch.no_grad()
def forecast_quantiles(model, state, K, n_paths=2000, seed=0, q=(0.05, 0.95)):
    """5-95% path quantiles of h_{t+k} (doc F4) from simulated log h paths."""
    lh = _sim_logh(model, state, K, n_paths, seed)
    return {"lo": np.exp(np.quantile(lh, q[0], 0)), "hi": np.exp(np.quantile(lh, q[1], 0))}


@torch.no_grad()
def term_structure(model, log_h, idx, Ks, periods_per_year=252):
    """Annualised sqrt((ppy/K) sum_{k<=K} E_t h_{t+k}) from low / median / high log h_{t+1}
    origins among idx (doc F5). For 5-min data h is in daily-equivalent units per bar,
    so the same formula annualises correctly."""
    lh = log_h.detach()
    sub = lh[idx]
    order = torch.argsort(sub)
    picks = {"low": int(idx[int(order[int(0.05 * (len(order) - 1))])]),
             "median": int(idx[int(order[len(order) // 2])]),
             "high": int(idx[int(order[int(0.95 * (len(order) - 1))])])}
    gb = model.g_bar if model.nn_active else 0.0
    res = {"K": list(Ks)}
    for nm, t in picks.items():
        h = model.forecast(lh[t: t + 1], max(Ks), g_bar=gb)["h"][0].cpu().numpy()
        cs = np.cumsum(h)
        res[nm] = {"t": t, "vol": [float(np.sqrt(periods_per_year / K * cs[K - 1])) for K in Ks]}
    return res


@torch.no_grad()
def nn_inputs(model, ds, out):
    """(T-1, n_in) inputs of g_t (t = 1..T-1): exo_{t-1} [, z_{t-1}, log h_{t-1}] and names."""
    f = ds.exo[:-1]
    names = [f"exo{j}" for j in range(f.shape[1])]
    nw = len(names) - 2
    names = [f"HAR{j}" for j in range(nw)] + ["sin cal", "cos cal"] if nw >= 0 else names
    if model.nn_state_inputs:
        X = torch.cat([f, out["z"][:-1, None], out["log_h"][:-1, None]], 1)
        names = names + ["z_{t-1}", "log h_{t-1}"]
    else:
        X = f
    return X, names


@torch.no_grad()
def g_from_inputs(model, X):
    if model.nn_state_inputs:
        return model.correction(X[:, :-2], X[:, -2], X[:, -1])
    return model.correction(X, X[:, 0] * 0, X[:, 0] * 0)


@torch.no_grad()
def partial_dependence(model, X, n_grid=15, n_rows=300, seed=0):
    """PD / ICE of g on each input (doc F7c): {j: {grid, mean, lo, hi}} (5-95% of ICE)."""
    rng = np.random.default_rng(seed)
    rows = X[torch.as_tensor(rng.choice(len(X), min(n_rows, len(X)), replace=False), device=X.device)]
    res = {}
    for j in range(X.shape[1]):
        xs = X[:, j].cpu().numpy()
        grid = np.quantile(xs, np.linspace(0.05, 0.95, n_grid))
        ice = []
        for gv in grid:
            Z = rows.clone()
            Z[:, j] = float(gv)
            ice.append(g_from_inputs(model, Z).cpu().numpy())
        ice = np.array(ice)  # (grid, rows)
        res[j] = {"grid": grid, "mean": ice.mean(1), "lo": np.quantile(ice, 0.05, 1),
                  "hi": np.quantile(ice, 0.95, 1)}
    return res


# ======================================================================== forecast evaluation
def cumulative_loss_diff(series_k, model="model", benches=("pure", "har", "garch11"), loss="qlike"):
    """Cumulative sum_{s<=t} (L^model_s - L^bench_s) over the test origins (doc E2)."""
    L = series_k["loss"]
    return {b: np.nancumsum(np.asarray(L[model][loss]) - np.asarray(L[b][loss])) for b in benches if b in L}


def mincer_zarnowitz(target, fc, k=1):
    """OLS x = a + b f with Newey-West s.e.; Wald test of (a, b) = (0, 1) ~ chi2(2); R^2 (doc E3)."""
    y, f = np.asarray(target, float), np.asarray(fc, float)
    ok = np.isfinite(y) & np.isfinite(f)
    y, f = y[ok], f[ok]
    X = np.column_stack([np.ones_like(f), f])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    lag = max(_nw_bandwidth(e), 2 * (k - 1), 1)
    V = _hac_cov(X, e, min(lag, len(e) - 1))
    dlt = beta - np.array([0.0, 1.0])
    W = float(dlt @ np.linalg.solve(V, dlt))
    r2 = 1 - e @ e / np.sum((y - y.mean()) ** 2)
    se = np.sqrt(np.diag(V))
    return {"a": float(beta[0]), "b": float(beta[1]), "se_a": float(se[0]), "se_b": float(se[1]),
            "wald": W, "wald_pvalue": chi2_sf(W, 2), "r2": float(r2), "n": int(len(y))}


def calibration_deciles(target, fc, n_bins=10):
    """Mean realized vs mean forecast in deciles of the forecast, +-2 s.e. (doc E5)."""
    y, f = np.asarray(target, float), np.asarray(fc, float)
    ok = np.isfinite(y) & np.isfinite(f)
    y, f = y[ok], f[ok]
    edges = np.quantile(f, np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.searchsorted(edges, f, side="right") - 1, 0, n_bins - 1)
    mf, my, se = [], [], []
    for i in range(n_bins):
        s = b == i
        if s.sum() == 0:
            continue
        mf.append(f[s].mean()), my.append(y[s].mean()), se.append(y[s].std() / math.sqrt(max(s.sum(), 1)))
    mf, my = np.array(mf), np.array(my)
    slope = float(np.polyfit(mf, my, 1)[0]) if len(mf) > 1 else float("nan")
    return {"mean_fc": mf.tolist(), "mean_x": my.tolist(), "se": list(map(float, se)), "slope": slope}


def model_confidence_set(losses, alpha=0.10, B=500, block=None, seed=0):
    """APPROXIMATE Model Confidence Set (Hansen, Lunde & Nason 2011), T_max statistic with a
    circular block bootstrap. losses: {name: (n,) per-t loss}. Returns MCS p-values and the set."""
    names = list(losses)
    L = np.column_stack([np.asarray(losses[n], float) for n in names])
    L = L[np.all(np.isfinite(L), 1)]
    n, m = L.shape
    if block is None:
        block = max(2, int(round(n ** (1 / 3))))
    rng = np.random.default_rng(seed)
    nb = int(np.ceil(n / block))
    boots = np.empty((B, m))
    for b in range(B):
        st = rng.integers(0, n, nb)
        idx = ((st[:, None] + np.arange(block)[None]) % n).ravel()[:n]
        boots[b] = L[idx].mean(0)
    mean = L.mean(0)
    alive = list(range(m))
    pv, prun = {}, 0.0
    while len(alive) > 1:
        a = np.array(alive)
        di = mean[a] - mean[a].mean()
        db = boots[:, a] - boots[:, a].mean(1, keepdims=True)
        var = np.mean((db - di) ** 2, 0)
        t = di / np.sqrt(np.maximum(var, 1e-300))
        tb = np.max((db - di) / np.sqrt(np.maximum(var, 1e-300)), 1)
        p = float(np.mean(tb >= t.max()))
        prun = max(prun, p)
        worst = alive[int(np.argmax(t))]
        pv[names[worst]] = prun
        alive.remove(worst)
    pv[names[alive[0]]] = 1.0
    return {"pvalues": pv, "mcs": [k for k, v in pv.items() if v >= alpha], "alpha": alpha,
            "B": B, "block": block, "approximate": True}


def kupiec(hits, alpha):
    """Unconditional coverage LR (Kupiec 1995) ~ chi2(1)."""
    h = np.asarray(hits, bool)
    n, x = len(h), int(h.sum())
    ph = x / n
    xl = lambda k, p: k * math.log(p) if k > 0 else 0.0
    lr = -2 * (xl(n - x, 1 - alpha) + xl(x, alpha) - xl(n - x, 1 - ph) - xl(x, ph))
    return {"n": n, "hits": x, "hit_rate": ph, "LR_uc": float(lr), "pvalue": chi2_sf(lr, 1)}


def christoffersen(hits, alpha):
    """Independence (first-order Markov) and conditional-coverage LRs (Christoffersen 1998)."""
    h = np.asarray(hits, int)
    a, b = h[:-1], h[1:]
    n00, n01 = int(np.sum((a == 0) & (b == 0))), int(np.sum((a == 0) & (b == 1)))
    n10, n11 = int(np.sum((a == 1) & (b == 0))), int(np.sum((a == 1) & (b == 1)))
    xl = lambda k, p: k * math.log(p) if k > 0 and p > 0 else 0.0
    p01 = n01 / max(n00 + n01, 1)
    p11 = n11 / max(n10 + n11, 1)
    p = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    l0 = xl(n00 + n10, 1 - p) + xl(n01 + n11, p)
    l1 = xl(n00, 1 - p01) + xl(n01, p01) + xl(n10, 1 - p11) + xl(n11, p11)
    lr_ind = float(max(-2 * (l0 - l1), 0.0))
    lr_uc = kupiec(h, alpha)["LR_uc"]
    lr_cc = lr_uc + lr_ind
    return {"LR_ind": lr_ind, "pvalue_ind": chi2_sf(lr_ind, 1), "LR_cc": lr_cc,
            "pvalue_cc": chi2_sf(lr_cc, 2)}


def var_backtest(r, mu, h, alphas=(0.01, 0.05)):
    """Gaussian VaR q_a = mu + sqrt(h) Phi^-1(a); hits r < q_a; Kupiec + Christoffersen (doc E6)."""
    r, h = np.asarray(r, float), np.asarray(h, float)
    res = {}
    for a in alphas:
        q = mu + np.sqrt(h) * float(norm_ppf(a))
        hits = r < q
        res[str(a)] = {**kupiec(hits, a), **christoffersen(hits, a), "q": q, "hits_mask": hits}
    return res


# ======================================================================== I3 / I4 (off by default)
def _stage1_cfg(cfg):
    return SimpleNamespace(stage1_streams=getattr(cfg, "stage1_streams", 1),
                           stage1_iters=getattr(cfg, "stage1_iters", 300),
                           stage1_burn_in=getattr(cfg, "stage1_burn_in", 22))


def parameter_recovery(variant, true_params, Ts=(500, 1000, 2500), n_rep=5, cfg=None, seed=0, with_se=True):
    """Doc I3: simulate the pure model at the true parameters, refit stage 1, collect estimates.
    Returns {"true": {...}, "T": {T: [params dict per replication]}, "se": {T: [sandwich s.e.
    dict per replication]}} ("se" only if with_se; used for the I3c z-scores). Off by default (slow)."""
    from .data import Dataset
    from .models import EGARCHPINN, GARCHPINN
    from .train import stage1_qmle
    cls = {"garch": GARCHPINN, "egarch": EGARCHPINN}[variant]
    c = _stage1_cfg(cfg)
    gen = cls(use_nn=False).load_params(**true_params)
    truth = {k: v for k, v in gen.params().items() if isinstance(v, float)}
    res = {"true": truth, "T": {}}
    if with_se:
        res["se"] = {}
    for T in Ts:
        est, ses = [], []
        for i in range(n_rep):
            r, lx, _ = gen.simulate(T, seed=seed + 1000 * i + T)
            ds = Dataset(r, lx, torch.zeros(T, 0, dtype=r.dtype, device=r.device), list(range(T)), T, 0)
            m = cls(use_nn=False)
            stage1_qmle(m, ds, c)
            est.append({k: v for k, v in m.params().items() if isinstance(v, float)})
            if with_se:
                se = standard_errors(m, ds, c.stage1_burn_in, T)
                ses.append(dict(zip(se["names"], se["se_sandwich"])))
        res["T"][T] = est
        if with_se:
            res["se"][T] = ses
    return res


def recovery_zscores(rec):
    """Doc I3c: {param: array of (estimate - truth) / sandwich s.e.} pooled over T and
    replications (equal to sqrt(T)(theta_hat - theta_0)/se of the sqrt(T)-scaled estimator).
    Non-finite or non-positive s.e. are dropped."""
    out = {}
    for T, ests in rec.get("T", {}).items():
        ses = rec.get("se", {}).get(T)
        if ses is None:
            continue
        for e, s in zip(ests, ses):
            for k, se in s.items():
                if k in e and k in rec["true"] and np.isfinite(se) and se > 0:
                    out.setdefault(k, []).append((e[k] - rec["true"][k]) / se)
    return {k: np.asarray(v) for k, v in out.items()}
    return res


def rolling_params(model, ds, cfg, n_windows=5, with_se=True, max_obs=6000):
    """Doc I4: expanding-window stage-1 refits ending at evenly spaced points of the sample.
    Returns a list of {end, params[, se]} (sandwich s.e. if with_se). Off by default."""
    from .data import Dataset
    from .train import stage1_qmle
    c = _stage1_cfg(cfg)
    ends = np.linspace(ds.n_train // 2, ds.T, n_windows).astype(int)
    out = []
    for e in ends:
        sub = Dataset(ds.r[:e], ds.log_x[:e], ds.exo[:e], ds.index[:e], int(e), 0)
        m = copy.deepcopy(model)
        stage1_qmle(m, sub, c)
        row = {"end": int(e), "params": {k: v for k, v in m.params().items() if isinstance(v, float)}}
        if with_se:
            se = standard_errors(m, sub, c.stage1_burn_in, max_obs)
            row["se"] = dict(zip(se["names"], se["se_sandwich"]))
        out.append(row)
    return out
