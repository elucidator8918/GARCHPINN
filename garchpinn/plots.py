"""Diagnostic figure for one fitted configuration (Agg backend, saved to disk).

All series are plotted on the model's return scale (percent for daily, bp for 5-min) as
volatility sqrt(h); nothing is annualised, so there is no risk of double annualisation.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

DPI = 80


def _np(v):
    try:
        return v.detach().cpu().numpy()
    except AttributeError:
        return np.asarray(v)


def _qlike_table(eval_metrics, key="qlike"):
    """{name: {k: QLIKE}} from evaluate()'s {"k=<k>": {name: {qlike, ...}}} output."""
    table = {}
    for hk, row in (eval_metrics or {}).items():
        if not (isinstance(hk, str) and hk.startswith("k=") and isinstance(row, dict)):
            continue
        k = int(hk[2:])
        for name, m in row.items():
            if isinstance(m, dict) and key in m:
                table.setdefault(name, {})[k] = float(m[key])
    return table


def shade_splits(a, ds, x=None, legend=True):
    """Shade the train / val / test periods (doc F1) on an axis indexed by t (or by x)."""
    x = np.arange(ds.T) if x is None else np.asarray(x)
    cuts = [0, ds.n_train, ds.n_train + ds.n_val, ds.T]
    for (lo, hi), c, lab in zip(zip(cuts[:-1], cuts[1:]), ("C2", "C1", "C3"), ("train", "val", "test")):
        if hi > lo:
            a.axvspan(x[lo], x[min(hi, len(x)) - 1], color=c, alpha=0.08, lw=0, label=lab if legend else None)


def plot_fit(name, ds, out, history, forecast, eval_metrics, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    fig, ax = plt.subplots(2, 2, figsize=(15, 9))

    # 1. training curves
    a = ax[0, 0]
    if history.get("train_loss"):
        a.plot(np.arange(1, len(history["train_loss"]) + 1), history["train_loss"], label="stage 2 train loss")
    if history.get("val_nll"):
        a.plot(np.arange(len(history["val_nll"])), history["val_nll"], label="stage 2 val NLL")
    for key, lab, ls in (("stage1_nll", "stage 1 train NLL", "--"), ("stage1_val_nll", "stage 1 val NLL", ":")):
        if key in history:
            a.axhline(history[key], color="k", ls=ls, lw=1, label=lab)
    if "best_epoch" in history:
        a.axvline(history["best_epoch"], color="gray", lw=0.8)
    a.set(title="Training (per-obs joint NLL)", xlabel="epoch")
    a.legend(fontsize=8)

    # 2. filtered sqrt h vs sqrt x
    a = ax[0, 1]
    idx = np.arange(ds.T)
    a.plot(idx, np.sqrt(np.exp(_np(ds.log_x))), lw=0.4, alpha=0.5, color="gray", label="sqrt x_t")
    a.plot(idx, np.sqrt(np.exp(_np(out["log_h"]))), lw=0.6, label="sqrt h_t (filtered)")
    shade_splits(a, ds)
    a.set(title="Filtered volatility vs realized measure", xlabel="t", yscale="log")
    a.legend(fontsize=8)

    # 3. forecast fan
    a = ax[1, 0]
    if forecast:
        h = _np(forecast["analytic"]["h"]).ravel()
        k = np.arange(1, len(h) + 1)
        a.plot(k, np.sqrt(h), label="analytic sqrt E h")
        if "log_h" in forecast["analytic"]:
            a.plot(k, np.sqrt(np.exp(_np(forecast["analytic"]["log_h"]).ravel())), "k--", lw=0.8,
                   label="naive exp(E log h)")
        if "mc" in forecast:
            a.plot(k, np.sqrt(_np(forecast["mc"]["h"]).ravel()), "--", label="Monte Carlo sqrt E h")
        if "q" in forecast:
            q = forecast["q"]
            a.fill_between(k, np.sqrt(_np(q["lo"])), np.sqrt(_np(q["hi"])), alpha=0.2, label="5-95% paths")
        if "pure" in forecast:
            a.plot(k, np.sqrt(_np(forecast["pure"]["h"]).ravel()), ":", label="pure model analytic")
    a.set(title="k-step forecast from last state", xlabel="horizon k")
    a.legend(fontsize=8)

    # 4. test QLIKE per horizon
    a = ax[1, 1]
    tab = _qlike_table(eval_metrics or {})
    if tab:
        hs = sorted({h for v in tab.values() for h in v})
        w = 0.8 / len(tab)
        for i, (m, v) in enumerate(tab.items()):
            a.bar(np.arange(len(hs)) + i * w, [v.get(h, np.nan) for h in hs], w, label=m)
        a.set_xticks(np.arange(len(hs)) + 0.4 - w / 2, [str(h) for h in hs])
        a.legend(fontsize=8)
    a.set(title="Test QLIKE by horizon (lower is better)", xlabel="horizon")

    fig.suptitle(name)
    fig.tight_layout()
    path = os.path.join(out_dir, f"{name}.png")
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


# ============================================================================ diagnostics figures
# One PNG per figure, named by the doc ID (docs/realized_garch_reference.tex, Section figs).
# Computations live in garchpinn/diagnostics.py; these functions only draw.
def _save(fig, out_dir, fname):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, fname)
    fig.tight_layout()
    fig.savefig(path, dpi=DPI)
    plt.close(fig)
    return path


def plot_F1(ds, out, out_dir, pure_out=None):
    fig, a = plt.subplots(figsize=(12, 4.5))
    idx = np.arange(ds.T)
    a.plot(idx, np.sqrt(np.exp(_np(ds.log_x))), lw=0.4, alpha=0.5, color="gray", label="sqrt x_t")
    if pure_out is not None:
        a.plot(idx, np.sqrt(np.exp(_np(pure_out["log_h"]))), lw=0.6, color="C0", alpha=0.7, label="sqrt h_t stage 1")
    a.plot(idx, np.sqrt(np.exp(_np(out["log_h"]))), lw=0.6, color="C3", label="sqrt h_t (model)")
    shade_splits(a, ds)
    a.set(yscale="log", xlabel="t", ylabel="volatility (return units)", title="F1 filtered volatility vs realized measure")
    a.legend(fontsize=8, ncol=3)
    return _save(fig, out_dir, "F1_filtered_vol.png")


def plot_F2(model, out, ds, out_dir, n_bins=20):
    import torch
    from .diagnostics import norm_ppf  # noqa: F401
    zg = torch.linspace(-4, 4, 201, dtype=ds.r.dtype)
    z = _np(out["z"]).ravel()
    lh = _np(out["log_h"]).ravel()
    resid = _np(ds.log_x) - float(model.xi) - float(model.phi) * lh  # = delta(z) + u (meas. eq.)
    order = np.argsort(z)
    bins = np.array_split(order, n_bins)
    bz = np.array([z[b].mean() for b in bins])
    bm = np.array([resid[b].mean() for b in bins])
    bs = np.array([resid[b].std() / np.sqrt(len(b)) for b in bins])
    with torch.no_grad():
        meas = _np(model.meas_lev(zg))
    eg = model.variant == "egarch"
    fig, ax = plt.subplots(1, 2 if eg else 1, figsize=(11 if eg else 6, 4.2), squeeze=False)
    a = ax[0, 0]
    zg_ = _np(zg)
    if eg:
        with torch.no_grad():
            tau = _np(model.t1 * zg + model.t2 * (zg * zg - 1))
        a.plot(zg_, tau, label="tau(z) (GARCH eq.)")
        a.set(xlabel="z", ylabel="log-variance units", title="F2 tau(z): news impact on log h_{t+1}")
        a.legend(fontsize=8)
        a = ax[0, 1]
        lab, ttl = "delta(z)", "F2 delta(z) (measurement eq.)"
    else:
        lab, ttl = "tau(z)", "F2 tau(z) (measurement eq.)"
        a.plot(zg_, float(model.gamma) * meas, ":", label="gamma tau(z): impact on log h_{t+1}")
    a.plot(zg_, meas, label=lab)
    a.errorbar(bz, bm - np.mean(bm) + np.mean(_np(model.meas_lev(torch.as_tensor(z)))), yerr=2 * bs,
               fmt="o", ms=3, color="k", lw=0.8, label="binned mean log x - xi - phi log h (+-2 se)")
    a.set(xlabel="z", ylabel="log-variance units", title=ttl, xlim=(-4, 4))
    a.legend(fontsize=7)
    return _save(fig, out_dir, "F2_news_impact.png")


def plot_F3(irf, out_dir, unit="days"):
    """Analytic IRF of E log h (a^(k-1) decay) with the half-life marked; MC overlay if present."""
    fig, a = plt.subplots(figsize=(7, 4.5))
    k = irf["k"]
    a.plot(k, irf["du"], lw=1.6, label="d E log h_{t+k} / d u_t (analytic, a^(k-1))")
    a.plot(k, irf["dz"], lw=1.0, label="d E log h_{t+k} / d z_t (linear part)")
    if "du_mc" in irf:
        a.plot(k, irf["du_mc"], ".", ms=2.5, alpha=0.5, color="gray", label="MC overlay (model incl. g)")
    hl = irf["half_life"]
    kh = irf.get("k_half", hl + 1)
    if np.isfinite(kh) and kh <= k[-1]:
        a.axvline(kh, color="gray", ls="--", lw=0.8)
        a.axhline(irf["du"][0] / 2, color="gray", ls=":", lw=0.6)
        a.annotate(f"half-life {hl:.1f} {unit}", (kh, irf["du"][0] / 2), xytext=(5, 8),
                   textcoords="offset points", fontsize=8)
    a.set(xlabel=f"horizon k ({unit})", ylabel="response",
          title=f"F3 impulse response, pi={irf['pi']:.4f}, half-life={hl:.1f} {unit}")
    a.legend(fontsize=8)
    ins = a.inset_axes([0.55, 0.35, 0.4, 0.3])
    ins.semilogy(k, np.abs(irf["du"]) + 1e-300)
    if np.isfinite(kh) and kh <= k[-1]:
        ins.axvline(kh, color="gray", ls="--", lw=0.6)
    ins.set_title("|du| log-y", fontsize=7)
    ins.tick_params(labelsize=6)
    return _save(fig, out_dir, "F3_impulse_response.png")


def plot_F4(fc, out_dir, q=None):
    fig, a = plt.subplots(figsize=(7, 4.5))
    h = _np(fc["analytic"]["h"]).ravel()
    k = np.arange(1, len(h) + 1)
    naive = np.sqrt(np.exp(_np(fc["analytic"]["log_h"]).ravel()))
    if q is not None:
        a.fill_between(k, np.sqrt(q["lo"]), np.sqrt(q["hi"]), alpha=0.15, label="5-95% paths")
    a.plot(k, np.sqrt(h), "C0", label="analytic sqrt E h (Jensen-corrected)")
    a.plot(k, naive, "k--", lw=1, label="naive sqrt exp(E log h)")
    a.fill_between(k, naive, np.sqrt(h), color="C1", alpha=0.3, label="Jensen term")
    if "mc" in fc:
        a.plot(k, np.sqrt(_np(fc["mc"]["h"]).ravel()), "r.", ms=4, label="Monte Carlo mean")
    if "pure" in fc:
        a.plot(k, np.sqrt(_np(fc["pure"]["h"]).ravel()), ":", label="pure model analytic")
    a.set(xlabel="horizon k", ylabel="sqrt E_t h_{t+k}", title="F4 k-step forecast and Jensen correction")
    a.legend(fontsize=8)
    return _save(fig, out_dir, "F4_forecast_jensen.png")


def plot_F5(ts, out_dir):
    fig, a = plt.subplots(figsize=(6.5, 4.5))
    for nm in ("low", "median", "high"):
        a.plot(ts["K"], ts[nm]["vol"], "o-", label=f"{nm}-vol start (t={ts[nm]['t']})")
    a.set(xlabel="horizon K", ylabel="annualised vol sqrt(252/K sum E h)", title="F5 term structure")
    a.legend(fontsize=8)
    return _save(fig, out_dir, "F5_term_structure.png")


def plot_F6(df, cfg, out_dir, n_boot=100, max_acf_obs=60000, seed=0):
    """Intraday periodicity (5-min only): needs df.attrs from data.fivemin_series."""
    from .diagnostics import acf
    from .data import periodicity_factor
    s2, s2_t, tr, bar = (df.attrs.get(k) for k in ("s2", "s2_t", "train_mask", "bar"))
    if s2 is None:
        return None
    r_f = df["r"].values
    r_raw = r_f * np.sqrt(s2_t)
    rng = np.random.default_rng(seed)
    days = df.index.normalize().values
    tdays = np.unique(days[tr])
    rt, bt, dt = r_raw[tr], bar[tr], days[tr]
    pos = {d: np.where(dt == d)[0] for d in tdays}
    boots = []
    for _ in range(n_boot):
        pick = np.concatenate([pos[d] for d in rng.choice(tdays, len(tdays))])
        boots.append(np.sqrt(periodicity_factor(rt[pick], bt[pick]).reindex(s2.index).values))
    boots = np.array(boots)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    a = ax[0]
    sk = np.sqrt(s2.values)
    a.fill_between(s2.index, np.nanquantile(boots, 0.025, 0), np.nanquantile(boots, 0.975, 0), alpha=0.3,
                   label="bootstrap 95% (days)")
    a.plot(s2.index, sk, label="s_k (train days)")
    a.set(xlabel="bar k (5-min since session open)", ylabel="s_k", title="F6a periodicity factor")
    a.legend(fontsize=8)
    a = ax[1]
    import pandas as pd
    a.plot(pd.Series(np.abs(r_raw)).groupby(bar).mean(), label="|r| raw")
    a.plot(pd.Series(np.abs(r_f)).groupby(bar).mean(), label="|r| / s_k")
    a.set(xlabel="bar k", ylabel="mean |r|", title="F6b mean |r| by bar")
    a.legend(fontsize=8)
    a = ax[2]
    nl = min(5 * int(np.median(pd.Series(bar).groupby(days).size())) + 5, len(r_f) // 2)
    sl = slice(-min(len(r_f), max_acf_obs), None)
    a.plot(acf(np.abs(r_raw[sl]), nl)[1:], lw=0.7, label="|r| raw")
    a.plot(acf(np.abs(r_f[sl]), nl)[1:], lw=0.7, label="|r| / s_k")
    a.set(xlabel="lag (bars)", ylabel="ACF", title="F6c ACF of |r| (5 days)")
    a.legend(fontsize=8)
    return _save(fig, out_dir, "F6_intraday_periodicity.png")


def plot_F7(ds, g, X, names, pd_res, lambda_g, out_dir):
    g = _np(g).ravel()
    fig, ax = plt.subplots(2, 2, figsize=(13, 8.5))
    a = ax[0, 0]
    a.plot(np.arange(len(g)), g, lw=0.5)
    m, s = g.mean(), g.std()
    a.axhline(m, color="k", lw=0.8)
    a.axhspan(m - 2 * s, m + 2 * s, color="gray", alpha=0.15, label="+-2 s.d.")
    shade_splits(a, ds)
    a.set(xlabel="t", ylabel="g_t", title="F7a NN correction g_t")
    a.legend(fontsize=7, ncol=4)
    a = ax[0, 1]
    a.hist(g, bins=60, color="C0", alpha=0.7)
    a.axvline(m, color="k", label=f"mean {m:.2e}")
    a.set(xlabel="g", title=f"F7b histogram of g (s.d. {s:.2e}; lambda_g={lambda_g:g} penalises mean g^2)")
    a.legend(fontsize=8)
    a = ax[1, 0]
    for j, nm in enumerate(names):
        v = pd_res[j]
        sd = float(np.std(_np(X[:, j]))) or 1.0
        gx = (v["grid"] - np.mean(_np(X[:, j]))) / sd
        a.plot(gx, v["mean"], label=nm)
        a.fill_between(gx, v["lo"], v["hi"], alpha=0.08)
    a.set(xlabel="input (standardised)", ylabel="g", title="F7c partial dependence (mean, 5-95% ICE)")
    a.legend(fontsize=7)
    a = ax[1, 1]
    if "log h_{t-1}" in names:
        x = _np(X[:, names.index("log h_{t-1}")])
        gg = g[1:]
        a.scatter(x, gg, s=2, alpha=0.3)
        b = np.polyfit(x, gg, 1)
        xs = np.linspace(x.min(), x.max(), 10)
        a.plot(xs, np.polyval(b, xs), "r", label=f"OLS slope {b[0]:.2e}")
        a.legend(fontsize=8)
    a.set(xlabel="log h_{t-1}", ylabel="g_t", title="F7d g vs log h_{t-1}")
    return _save(fig, out_dir, "F7_nn_correction.png")


def _splits(ds):
    return {"train": ds.split("train"), "test": ds.split("test")}


def plot_R1(z, u, ds, stats, out_dir):
    from .diagnostics import norm_ppf
    fig, ax = plt.subplots(2, 4, figsize=(16, 7.5))
    for i, (sp, sl) in enumerate(_splits(ds).items()):
        for j, (nm, v) in enumerate((("z", z[sl]), ("u~", u[sl]))):
            v = np.sort(v[np.isfinite(v)])
            q = norm_ppf((np.arange(1, len(v) + 1) - 0.5) / len(v))
            jb = stats[sp]["jb"]["z" if nm == "z" else "u"]
            a = ax[i, 2 * j]
            a.plot(q, v, ".", ms=2)
            a.plot(q, q, "r", lw=0.8)
            a.set(title=f"R1 QQ {nm} ({sp}) JB={jb['JB']:.1f} p={jb['pvalue']:.3f}", xlabel="N(0,1)")
            a = ax[i, 2 * j + 1]
            a.hist(v, bins=60, density=True, alpha=0.6)
            xs = np.linspace(-4, 4, 200)
            a.plot(xs, np.exp(-xs ** 2 / 2) / np.sqrt(2 * np.pi), "r", lw=0.8)
            a.set(title=f"R1 hist {nm} ({sp}) skew={jb['skew']:.2f} kurt={jb['kurt']:.2f}", xlim=(-5, 5))
    return _save(fig, out_dir, "R1_qq_hist.png")


def plot_R2(z, u, ds, stats, nlags, out_dir):
    from .diagnostics import acf
    names = (("z", lambda z, u: z), ("z2", lambda z, u: z * z), ("u", lambda z, u: u),
             ("u2", lambda z, u: u * u), ("zu", lambda z, u: z * u))
    fig, ax = plt.subplots(2, 5, figsize=(18, 6.5))
    for i, (sp, sl) in enumerate(_splits(ds).items()):
        for j, (nm, fn) in enumerate(names):
            v = fn(z[sl], u[sl])
            n = np.isfinite(v).sum()
            nl = min(nlags, n // 2)
            a = ax[i, j]
            a.bar(np.arange(1, nl + 1), acf(v, nl)[1:], width=1.0 if nl > 60 else 0.6)
            b = 1.96 / np.sqrt(n)
            a.axhline(b, color="r", ls="--", lw=0.8), a.axhline(-b, color="r", ls="--", lw=0.8)
            lb = stats[sp]["ljung_box"][nm]
            txt = ", ".join(f"Q({m}) p={v_['pvalue']:.3f}" for m, v_ in lb.items())
            a.set_title(f"R2 ACF {nm} ({sp})\n{txt}", fontsize=8)
    return _save(fig, out_dir, "R2_acf_ljung_box.png")


def plot_R3(mpaths, ds, out_dir):
    from .diagnostics import running_mean_band
    k = len(mpaths)
    nc = 4
    nr = int(np.ceil(k / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(16, 2.8 * nr), squeeze=False)
    for i, (nm, m) in enumerate(mpaths.items()):
        a = ax[i // nc, i % nc]
        rm, band = running_mean_band(m)
        t = np.arange(len(rm))
        a.plot(t, rm, lw=0.8)
        a.fill_between(t, -band, band, alpha=0.2, color="gray")
        a.axhline(0, color="k", lw=0.5)
        a.set_ylim(-4 * np.nanmedian(band), 4 * np.nanmedian(band))
        shade_splits(a, ds, legend=False)
        a.set_title(f"R3 running mean {nm}", fontsize=9)
    for j in range(k, nr * nc):
        ax[j // nc, j % nc].axis("off")
    return _save(fig, out_dir, "R3_moment_residuals.png")


def plot_R4(z, ds, out_dir, nb=20):
    from .diagnostics import norm_cdf
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    for i, (sp, sl) in enumerate(_splits(ds).items()):
        p = norm_cdf(z[sl][np.isfinite(z[sl])])
        n = len(p)
        a = ax[i, 0]
        a.hist(p, bins=nb, range=(0, 1), alpha=0.7)
        e = n / nb
        sd = np.sqrt(n * (1 / nb) * (1 - 1 / nb))
        a.axhspan(e - 1.96 * sd, e + 1.96 * sd, color="r", alpha=0.15, label="95% binomial")
        a.axhline(e, color="r", lw=0.8)
        a.set(title=f"R4 PIT histogram ({sp})", xlabel="Phi(z)")
        a.legend(fontsize=8)
        a = ax[i, 1]
        ps = np.sort(p)
        a.plot(ps, np.arange(1, n + 1) / n, lw=0.8)
        a.plot([0, 1], [0, 1], "r", lw=0.6)
        ks = 1.36 / np.sqrt(n)
        a.fill_between([0, 1], [-ks, 1 - ks], [ks, 1 + ks], color="gray", alpha=0.15, label="95% KS band")
        a.set(title=f"R4 ECDF of PIT ({sp})", xlim=(0, 1), ylim=(0, 1))
        a.legend(fontsize=8)
    return _save(fig, out_dir, "R4_pit.png")


def plot_R5(z, u, ds, out_dir, max_lag=10):
    from .diagnostics import cross_corr
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for a, (sp, sl) in zip(ax, _splits(ds).items()):
        js, cc = cross_corr(z[sl], u[sl], max_lag)
        a.bar(js, cc, width=0.6)
        b = 1.96 / np.sqrt(len(z[sl]))
        a.axhline(b, color="r", ls="--", lw=0.8), a.axhline(-b, color="r", ls="--", lw=0.8)
        a.set(title=f"R5 corr(z_t, u~_t+j) ({sp}); j=0: {cc[max_lag]:.3f}", xlabel="j")
    return _save(fig, out_dir, "R5_cross_corr.png")


def plot_I1(prof, out_dir):
    P = prof["profiles"]
    fig, ax = plt.subplots(1, max(len(P), 1), figsize=(4 * max(len(P), 1), 3.8), squeeze=False)
    for a, (nm, v) in zip(ax[0], P.items()):
        a.plot(v["grid"], v["lr"], "o-", ms=3)
        a.axhline(prof["cut"], color="r", ls="--", lw=0.8, label="chi2_1(0.95)")
        a.axvline(v["estimate"], color="gray", lw=0.6)
        a.set(xlabel=nm, ylabel="2 (l_max - l_p)", title=f"I1 profile {nm}")
        a.legend(fontsize=7)
    return _save(fig, out_dir, "I1_profile_likelihood.png")


def plot_I2(se, out_dir):
    n = len(se["names"])
    fig, a = plt.subplots(figsize=(7, 0.45 * n + 1.5))
    y = np.arange(n)
    est, sh, ss = (np.array(se[k]) for k in ("estimate", "se_hessian", "se_sandwich"))
    a.errorbar(est, y + 0.12, xerr=1.96 * sh, fmt="o", ms=3, label="Hessian 95%")
    a.errorbar(est, y - 0.12, xerr=1.96 * ss, fmt="s", ms=3, label="sandwich 95%")
    a.set_yticks(y, [f"{nm} (ratio {r:.2f})" for nm, r in zip(se["names"], se["ratio"])])
    a.axvline(0, color="k", lw=0.4)
    a.set(title=f"I2 stage-1 estimates, n={se['n']}", xlabel="estimate")
    a.legend(fontsize=8)
    return _save(fig, out_dir, "I2_standard_errors.png")


def plot_I3(rec, out_dir):
    truth = rec["true"]
    Ts = sorted(rec["T"])
    keys = [k for k in truth if all(k in e for e in rec["T"][Ts[0]]) and k not in ("half_life", "e_log_abs_A")]
    from .diagnostics import norm_ppf, recovery_zscores
    zs = recovery_zscores(rec)
    fig, ax = plt.subplots(1, 4 if zs else 2, figsize=(24 if zs else 12, 4.5))
    a = ax[0]
    for k in keys:
        est = [e[k] for e in rec["T"][Ts[-1]]]
        a.plot([truth[k]] * len(est), est, "o", ms=3, label=k)
    lim = [min(truth.values()) - 0.1, max(v for v in truth.values()) + 0.1]
    a.plot(lim, lim, "k", lw=0.6)
    a.set(xlabel="truth", ylabel=f"estimate (T={Ts[-1]})", title="I3a estimate vs truth")
    a.legend(fontsize=6, ncol=2)
    a = ax[1]
    for k in keys:
        rmse = [np.sqrt(np.mean([(e[k] - truth[k]) ** 2 for e in rec["T"][T]])) for T in Ts]
        a.loglog(Ts, rmse, "o-", ms=3, label=k)
    a.loglog(Ts, 0.5 * (np.array(Ts) / Ts[0]) ** -0.5, "k--", lw=0.6, label="slope -1/2")
    a.set(xlabel="T", ylabel="RMSE", title="I3b RMSE vs T")
    a.legend(fontsize=6, ncol=2)
    if zs:
        a = ax[2]
        allz = np.concatenate(list(zs.values()))
        allz = allz[np.abs(allz) < 10]
        a.hist(allz, bins=30, range=(-5, 5), density=True, alpha=0.6, label="pooled z")
        g = np.linspace(-5, 5, 200)
        a.plot(g, np.exp(-g ** 2 / 2) / np.sqrt(2 * np.pi), "r", lw=0.8, label="N(0,1)")
        a.set(xlabel="(estimate - truth) / se", title=f"I3c z-scores, n={len(allz)}")
        a.legend(fontsize=6)
        a = ax[3]
        for k, v in zs.items():
            v = np.sort(v)
            n = len(v)
            q = np.array([norm_ppf((i + 0.5) / n) for i in range(n)])
            a.plot(q, v, "o", ms=3, label=k)
        a.plot([-3, 3], [-3, 3], "k", lw=0.6)
        a.set(xlabel="N(0,1) quantile", ylabel="z", title="I3c QQ of z-scores", ylim=(-6, 6))
        a.legend(fontsize=6, ncol=2)
    return _save(fig, out_dir, "I3_parameter_recovery.png")


def plot_I4(roll, out_dir, keys=("persistence", "phi", "gamma", "xi", "sigma_u")):
    fig, ax = plt.subplots(1, len(keys), figsize=(3.4 * len(keys), 3.5))
    ends = [r["end"] for r in roll]
    for a, k in zip(ax, keys):
        v = np.array([r["params"].get(k, np.nan) for r in roll])
        a.plot(ends, v, "o-", ms=3)
        if all("se" in r for r in roll):
            s = np.array([r["se"].get(k, np.nan) for r in roll])
            a.fill_between(ends, v - 2 * s, v + 2 * s, alpha=0.2)
        a.set(title=f"I4 {k}", xlabel="window end t")
    return _save(fig, out_dir, "I4_rolling_params.png")


def plot_I5(st, out_dir, pinn_point=None, cloud=None):
    fig, a = plt.subplots(figsize=(5.5, 5))
    x, y = st["x"], st["y"]
    if st["variant"] == "garch":
        xs = np.linspace(-0.2, 1.2, 50)
        a.plot(xs, 1 - xs, "k", lw=1, label="beta + gamma phi = 1")
        a.fill_between(xs, 1 - xs, 1.5, color="gray", alpha=0.2)
        a.set(xlabel="beta", ylabel="gamma phi")
    else:
        for b in (-1, 1):
            a.axvline(b, color="k", lw=1)
        a.set(xlabel="beta", ylabel="gamma")
        a.text(1.0, y, " |beta|=1", fontsize=8)
    if "cov" in st:
        C = np.array(st["cov"])
        w, V = np.linalg.eigh(C)
        t = np.linspace(0, 2 * np.pi, 100)
        circ = np.stack([np.cos(t), np.sin(t)])
        e = V @ (np.sqrt(np.maximum(w, 0) * 5.991)[:, None] * circ)
        a.plot(x + e[0], y + e[1], "C0", label="95% sandwich ellipse")
    if cloud is not None and len(cloud):
        c = np.array(cloud)
        a.plot(c[:, 0], c[:, 1], ".", color="C2", label="rolling estimates")
    a.plot(x, y, "o", color="C0", label="stage-1 estimate")
    if pinn_point is not None:
        a.plot(*pinn_point, "o", color="C3", label="stage 2 pinn_persistence")
    # limits cover the estimate, the ellipse, the cloud and the stage-2 point (plus a margin)
    pts = [np.array([[x, y]])]
    if "cov" in st:
        pts.append(np.stack([x + e[0], y + e[1]], 1))
    if cloud is not None and len(cloud):
        pts.append(np.asarray(cloud, float).reshape(-1, 2))
    if pinn_point is not None:
        pts.append(np.asarray(pinn_point, float).reshape(1, 2))
    P_ = np.concatenate(pts)
    P_ = P_[np.all(np.isfinite(P_), 1)]
    lo, hi = P_.min(0), P_.max(0)
    mx, my = 0.1 * max(hi[0] - lo[0], 0.1), 0.1 * max(hi[1] - lo[1], 0.1)
    a.set_xlim(min(lo[0] - mx, 0), max(hi[0] + mx, 1.05))
    if st["variant"] == "garch":
        a.set_ylim(min(lo[1] - my, 0), max(hi[1] + my, 1.05))
    else:
        a.set_ylim(lo[1] - my, hi[1] + my)
    a.set_title("I5 stationarity region")
    a.legend(fontsize=7)
    return _save(fig, out_dir, "I5_stationarity.png")


def plot_E2(cum, dm, out_dir):
    ks = list(cum)
    fig, ax = plt.subplots(1, len(ks), figsize=(5 * len(ks), 4), squeeze=False)
    for a, hk in zip(ax[0], ks):
        for b, c in cum[hk].items():
            st = dm.get(hk, {}).get(f"model_vs_{b}", {}).get("stat", float("nan"))
            a.plot(c, lw=0.8, label=f"vs {b} (DM={st:.2f})")
        a.axhline(0, color="k", lw=0.5)
        a.set(title=f"E2 cumulative QLIKE diff, {hk}", xlabel="test origin")
        a.legend(fontsize=7)
    return _save(fig, out_dir, "E2_dm_path.png")


def plot_E3(series, mz, out_dir, name="model"):
    ks = list(series)
    fig, ax = plt.subplots(1, len(ks), figsize=(5 * len(ks), 4.3), squeeze=False)
    for a, hk in zip(ax[0], ks):
        f, y = np.asarray(series[hk]["forecast"][name]), np.asarray(series[hk]["target"])
        a.loglog(f, y, ".", ms=2, alpha=0.4)
        lo, hi = np.nanmin(f), np.nanmax(f)
        xs = np.geomspace(lo, hi, 50)
        r = mz[hk]
        fit = r["a"] + r["b"] * xs
        a.plot(xs[fit > 0], fit[fit > 0], "r", label=f"a={r['a']:.2f}({r['se_a']:.2f}) b={r['b']:.2f}({r['se_b']:.2f})")
        a.plot(xs, xs, "k--", lw=0.6, label="45 deg")
        a.set(title=f"E3 MZ {hk}: Wald p={r['wald_pvalue']:.3f}, R2={r['r2']:.2f}", xlabel="forecast", ylabel="realized")
        a.legend(fontsize=7)
    return _save(fig, out_dir, "E3_mincer_zarnowitz.png")


def plot_E4(mcs, out_dir):
    hks = list(mcs)
    models = sorted({m for v in mcs.values() for m in v["pvalues"]})
    P = np.array([[mcs[h]["pvalues"].get(m, np.nan) for h in hks] for m in models])
    fig, a = plt.subplots(figsize=(1.6 * len(hks) + 3, 0.5 * len(models) + 1.5))
    inset = np.array([[m in mcs[h]["mcs"] for h in hks] for m in models], float)
    a.imshow(inset, cmap="Blues", vmin=0, vmax=1.6, aspect="auto")
    for i in range(len(models)):
        for j in range(len(hks)):
            a.text(j, i, f"{P[i, j]:.2f}", ha="center", va="center", fontsize=8)
    a.set_xticks(range(len(hks)), hks)
    a.set_yticks(range(len(models)), models)
    al = mcs[hks[0]]["alpha"]
    a.set_title(f"E4 {int((1 - al) * 100)}% MCS (shaded), MCS p-values [approximate]", fontsize=9)
    return _save(fig, out_dir, "E4_mcs.png")


def plot_E5(cal, out_dir):
    ks = list(cal)
    fig, ax = plt.subplots(1, len(ks), figsize=(4.5 * len(ks), 4.2), squeeze=False)
    for a, hk in zip(ax[0], ks):
        c = cal[hk]
        a.errorbar(c["mean_fc"], c["mean_x"], yerr=2 * np.array(c["se"]), fmt="o", ms=3)
        lo, hi = min(c["mean_fc"] + c["mean_x"]), max(c["mean_fc"] + c["mean_x"])
        a.plot([lo, hi], [lo, hi], "k--", lw=0.6)
        a.set(xscale="log", yscale="log", title=f"E5 calibration {hk} (slope {c['slope']:.2f})",
              xlabel="mean forecast (decile)", ylabel="mean realized")
    return _save(fig, out_dir, "E5_calibration.png")


def plot_E6(r, var, out_dir):
    fig, a = plt.subplots(figsize=(12, 4.2))
    t = np.arange(len(r))
    a.plot(t, r, lw=0.4, color="gray", label="r_t (test)")
    for i, (al, v) in enumerate(var.items()):
        a.plot(t, v["q"], lw=0.7, color=f"C{i}",
               label=f"VaR {al}: hit {v['hit_rate']:.3f}, Kupiec p={v['pvalue']:.2f}, CC p={v['pvalue_cc']:.2f}")
        h = np.asarray(v["hits_mask"])
        a.plot(t[h], np.asarray(r)[h], "x", ms=4, color=f"C{i}")
    a.set(title="E6 Gaussian VaR backtest", xlabel="test t")
    a.legend(fontsize=7)
    return _save(fig, out_dir, "E6_var_backtest.png")
