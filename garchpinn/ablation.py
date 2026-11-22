"""Ablation driver (reference doc, section "Ablation study design", plots P1-P5).

Each variant is a Config override (plus an optional dataset transform). Every run fits the
two-stage model with main.fit_two_stage, evaluates it with main.run_evaluate and appends one
tidy row to <out>/runs.jsonl (source of truth; resumable), from which results.csv/json are
rebuilt. Per-t test QLIKE losses (one row per horizon, NaN padded) go to <out>/losses/*.npy
as float32 for the paired DM matrix (P4).
"""
import itertools
import json
import math
import os
import time

import numpy as np
import pandas as pd
import torch

from .config import Config
from .data import build_dataset, get_series
from .evaluate import _model_sum_forecasts, _t_sf2, dm_test, qlike
from .models import build_model
from .utils import _jsonable, set_seeds


def _drop_har(ds, cfg):
    """Remove the HAR log-x averages from the NN inputs (keep the calendar features)."""
    k = len(cfg.har_windows)
    ds.exo = ds.exo[..., k:].contiguous()
    if ds.exo_mu is not None:
        ds.exo_mu, ds.exo_sd = ds.exo_mu[k:], ds.exo_sd[k:]
    return ds


# name -> (table row, Config overrides, dataset transform)
VARIANTS = {
    "baseline": ("default Config", {}, None),
    "no_nn": ("NN on vs off (stage 2 vs 1)", {"use_nn": False}, None),
    "lambda_m0": ("lambda_m = 0", {"lambda_moment": 0.0}, None),
    "lambda_g0": ("lambda_g = 0", {"lambda_g": 0.0}, None),
    "lambda_orth0": ("lambda_orth = 0", {"lambda_orth": 0.0}, None),
    "no_state": ("no state inputs (z, log h)", {"nn_state_inputs": False}, None),
    "no_har": ("HAR features removed", {}, _drop_har),
    "hidden8": ("hidden size 8", {"hidden_dim": 8}, None),
    "hidden128": ("hidden size 128", {"hidden_dim": 128}, None),
    "chunk32": ("chunk length 32", {"chunk_len": 32}, None),
    "chunk512": ("chunk length 512", {"chunk_len": 512}, None),
    "reinit_state": ("re-initialised state per chunk", {"carry_state": False}, None),
    "tau0": ("tau == 0 (no leverage)", {"no_leverage": True}, None),
    "partial_lik": ("no measurement eq. (partial l_r)", {"likelihood": "partial"}, None),
}
# Table rows covered by the run axes rather than a Config override: compare models / freqs
# of the "baseline" rows (P1 groups by model and freq).
AXIS_ROWS = {"garch vs egarch": "--models garch,egarch", "daily vs 5-min": "--freqs daily,5min",
             "hidden 32 / chunk 128": "baseline"}

GRID = {"lambda_moment": (0.0, 0.3, 1.0), "lambda_g": (0.0, 1e-2, 1e-1), "lambda_orth": (0.0, 1.0)}


def grid_variants():
    out = {}
    for lm, lg, lo in itertools.product(*GRID.values()):
        out[f"grid_lm{lm:g}_lg{lg:g}_lo{lo:g}"] = (
            "lambda grid", {"lambda_moment": lm, "lambda_g": lg, "lambda_orth": lo}, None)
    return out


def all_variants(grid=False):
    return {**VARIANTS, **(grid_variants() if grid else {})}


def run_id(variant, model, freq, seed):
    return f"{variant}|{model}|{freq}|s{seed}"


def make_config(variant, model, freq, seed, base=None, variants=None):
    """Config for one run. Overrides are applied after __post_init__ so the 5-min defaults
    (e.g. chunk_len >= 512) do not undo them."""
    base = dict(base or {})
    epochs = base.pop("epochs", None)
    cfg = Config(model=model, freq=freq, seed=seed, **base)
    if epochs is not None:
        cfg.epochs = epochs
    for k, v in (variants or VARIANTS)[variant][1].items():
        setattr(cfg, k, v)
    return cfg


def per_t_qlike(model, ds, cfg):
    """(K, n) float32 per-origin test QLIKE of the k-sum forecasts, same origins and targets
    as evaluate.evaluate; rows NaN padded to the k=1 length."""
    with torch.no_grad():
        _, fc = _model_sum_forecasts(model, ds, tuple(cfg.horizons))
    x = np.exp(ds.log_x.detach().cpu().numpy())
    csx = np.concatenate([[0.0], np.cumsum(x)])
    t0, T = ds.split("test").start, ds.T
    rows = []
    for k in cfg.horizons:
        t = np.arange(t0 - 1, T - k)
        rows.append(qlike(csx[t + k + 1] - csx[t + 1], fc[k][t]))
    n = max(len(r) for r in rows)
    out = np.full((len(rows), n), np.nan, np.float32)
    for i, r in enumerate(rows):
        out[i, : len(r)] = r
    return out


def run_one(cfg, df, transform=None):
    """Fit + evaluate one configuration; returns (row dict, history, per-t losses)."""
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if root not in sys.path:
        sys.path.insert(0, root)
    import main as M  # repo-root main.py: reuse its two-stage fit and evaluation

    set_seeds(cfg.seed)
    ds = build_dataset(df, cfg)
    if transform is not None:
        ds = transform(ds, cfg)
    model = build_model(cfg, ds.exo.shape[-1])
    t_start = time.time()
    hist, pure = M.fit_two_stage(model, ds, cfg)
    model.eval()
    from .train import val_nll
    vj = float(val_nll(model, ds))
    metrics = M.run_evaluate(model, pure, ds, cfg)
    losses = per_t_qlike(model, ds, cfg)
    row = {"val_nll": vj, "val_nll_objective": hist.get("best_val_nll", hist["stage1_val_nll"]),
           "stage1_val_nll": hist["stage1_val_nll"], "stage1_nll": hist["stage1_nll"],
           "best_epoch": hist.get("best_epoch", 0), "epochs_run": len(hist.get("train_loss", []))}
    ll = metrics["loglik"]
    row["test_l_r"], row["test_l_rx"] = ll["model"]["l_r"], ll["model"]["l_rx"]
    row["pure_test_l_r"], row["pure_test_l_rx"] = ll["pure"]["l_r"], ll["pure"]["l_rx"]
    for i, k in enumerate(cfg.horizons):
        mk = metrics[f"k={k}"]
        row[f"h{i}"] = k
        for m in ("qlike", "mse"):
            row[f"{m}_h{i}"] = mk["model"][m]
        row[f"pure_qlike_h{i}"] = mk["pure"]["qlike"]
    for k, v in model.params().items():
        if isinstance(v, (int, float, bool)):
            row[f"p_{k}"] = float(v)
    for k, v in pure.params().items():
        if isinstance(v, (int, float, bool)):
            row[f"s1_{k}"] = float(v)
    row["runtime_s"] = time.time() - t_start
    hist_small = {"train_loss": hist.get("train_loss", []), "val_nll": hist.get("val_nll", [])}
    return row, hist_small, losses


def _load_done(out):
    p = os.path.join(out, "runs.jsonl")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def write_tables(rows, out):
    flat = [{k: v for k, v in r.items() if k != "history"} for r in rows]
    df = pd.DataFrame(flat)
    df.to_csv(os.path.join(out, "results.csv"), index=False)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(_jsonable(rows), f, indent=1, default=str)
    return df


def run_ablation(variants, models, freqs, seeds, out, data_path=None, epochs=None,
                 base=None, grid=False, verbose=True):
    """Run every variant x model x freq x seed not already in <out>/runs.jsonl."""
    os.makedirs(os.path.join(out, "losses"), exist_ok=True)
    reg = all_variants(grid)
    base = dict(base or {})
    if data_path:
        base["data_path"] = data_path
    if epochs is not None:
        base["epochs"] = epochs
    rows = _load_done(out)
    done = {r["run_id"] for r in rows}
    cache = {}
    for fq in freqs:
        for v in variants:
            for m in models:
                for s in seeds:
                    rid = run_id(v, m, fq, s)
                    if rid in done:
                        continue
                    cfg = make_config(v, m, fq, s, base, reg)
                    if fq not in cache:
                        cache[fq] = get_series(cfg)
                    row, hist, losses = run_one(cfg, cache[fq], reg[v][2])
                    fn = rid.replace("|", "__") + ".npy"
                    np.save(os.path.join(out, "losses", fn), losses)
                    row = {"run_id": rid, "variant": v, "model": m, "freq": fq, "seed": s,
                           "lambda_moment": cfg.lambda_moment, "lambda_g": cfg.lambda_g,
                           "lambda_orth": cfg.lambda_orth, "losses_file": fn, **row, "history": hist}
                    with open(os.path.join(out, "runs.jsonl"), "a") as f:
                        f.write(json.dumps(_jsonable(row), default=str) + "\n")
                    rows.append(row)
                    done.add(rid)
                    if verbose:
                        print(f"[{len(done)}] {rid}: test l_rx={row['test_l_rx']:.4f} "
                              f"qlike_h0={row['qlike_h0']:.4f} ({row['runtime_s']:.1f}s)", flush=True)
    return write_tables(rows, out), rows


# ------------------------------------------------------------------ statistics
def tci(x):
    """mean, sd, 95% t half-width."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return np.nan, np.nan, np.nan
    if n == 1:
        return x[0], 0.0, 0.0
    sd = x.std(ddof=1)
    return x.mean(), sd, t975(n - 1) * sd / math.sqrt(n)


def t975(df):
    """0.975 quantile of Student t (bisection on evaluate's two-sided t tail; no scipy)."""
    lo, hi = 0.0, 100.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if _t_sf2(mid, df) > 0.05 else (lo, mid)
    return 0.5 * (lo + hi)


def holm(p):
    p = np.asarray(p, float)
    adj = np.full_like(p, np.nan)
    ok = np.where(np.isfinite(p))[0]
    order = ok[np.argsort(p[ok])]
    m, run = len(order), 0.0
    for i, j in enumerate(order):
        run = max(run, min(1.0, (m - i) * p[j]))
        adj[j] = run
    return adj


def dm_matrix(rows, out, model, freq, hi=0):
    """Paired DM over variants on seed-averaged per-t QLIKE at horizon index hi.
    Returns (names, stat, holm-adjusted p). stat < 0: row variant better than column."""
    by = {}
    for r in rows:
        if r["model"] == model and r["freq"] == freq:
            a = np.load(os.path.join(out, "losses", r["losses_file"]))[hi]
            by.setdefault(r["variant"], []).append(a[np.isfinite(a)].astype(float))
    names = list(by)
    L = {}
    for v, arrs in by.items():
        n = min(len(a) for a in arrs)
        L[v] = np.mean([a[:n] for a in arrs], 0)
    k = int(next(r[f"h{hi}"] for r in rows if r["model"] == model and r["freq"] == freq))
    V = len(names)
    S, P = np.full((V, V), np.nan), np.full((V, V), np.nan)
    for i, j in itertools.combinations(range(V), 2):
        a, b = L[names[i]], L[names[j]]
        n = min(len(a), len(b))
        d = dm_test(a[:n], b[:n], k=k)
        S[i, j], S[j, i] = d["stat"], -d["stat"]
        P[i, j] = d["pvalue"]
    iu = np.triu_indices(V, 1)
    P[iu] = holm(P[iu])
    P.T[iu] = P[iu]
    return names, S, P


def summary_table(df, metrics=("test_l_r", "test_l_rx", "qlike_h0", "qlike_h1", "qlike_h2", "val_nll")):
    """Markdown table: mean +- 95% t half-width over seeds per variant x model x freq."""
    metrics = [m for m in metrics if m in df]
    hdr = "| variant | model | freq | n | " + " | ".join(metrics) + " | runtime_s |"
    lines = [hdr, "|" + "---|" * (5 + len(metrics))]
    for (v, m, f), g in df.groupby(["variant", "model", "freq"], sort=False):
        cells = []
        for c in metrics:
            mu, _, hw = tci(g[c])
            cells.append(f"{mu:.4f} ± {hw:.4f}")
        lines.append(f"| {v} | {m} | {f} | {len(g)} | " + " | ".join(cells)
                     + f" | {g['runtime_s'].mean():.1f} |")
    return "\n".join(lines)


# ------------------------------------------------------------------ plots
def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _cells(df):
    return list(df.groupby(["model", "freq"], sort=False).groups)


def plot_p1(df, out, dpi=120):
    """Grouped bars of test NLL (-l_r, -l_rx) and QLIKE per horizon, mean with 95% seed CI."""
    plt = _plt()
    mets = [c for c in ("test_l_r", "test_l_rx", "qlike_h0", "qlike_h1", "qlike_h2") if c in df]
    sign = {"test_l_r": -1, "test_l_rx": -1}
    cells = _cells(df)
    variants = list(dict.fromkeys(df["variant"]))
    fig, axes = plt.subplots(len(mets), 1, figsize=(max(6, 0.6 * len(variants) * len(cells)), 2.6 * len(mets)),
                             squeeze=False)
    w = 0.8 / max(len(cells), 1)
    for ax, met in zip(axes[:, 0], mets):
        for ci, (m, f) in enumerate(cells):
            g = df[(df.model == m) & (df.freq == f)]
            st = [tci(sign.get(met, 1) * g[g.variant == v][met]) for v in variants]
            mu, hw = np.array([s[0] for s in st]), np.array([s[2] for s in st])
            ax.bar(np.arange(len(variants)) + ci * w, mu, w, yerr=hw, capsize=2, label=f"{m}/{f}")
        lab = {"test_l_r": "test NLL r", "test_l_rx": "test NLL r,x"}.get(met, met)
        if met.startswith("qlike_h"):
            hs = df[met.replace("qlike_h", "h")].dropna()
            lab = f"QLIKE k={int(hs.iloc[0])}" if len(hs) else met
        ax.set_ylabel(lab)
        lo = np.nanmin([b.get_height() for b in ax.patches]) if ax.patches else 0
        hi_ = np.nanmax([b.get_height() for b in ax.patches]) if ax.patches else 1
        pad = 0.15 * (hi_ - lo + 1e-9)
        ax.set_ylim(lo - pad, hi_ + pad)
        ax.set_xticks(np.arange(len(variants)) + w * (len(cells) - 1) / 2)
        ax.set_xticklabels(variants, rotation=40, ha="right", fontsize=8)
    axes[0, 0].legend(fontsize=7)
    axes[0, 0].set_title("P1: test NLL and QLIKE (mean, 95% seed CI); lower is better")
    fig.tight_layout()
    p = os.path.join(out, "P1_bars.png")
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return p


def plot_p2(df, out, dpi=120):
    """Val NLL heatmap over (lambda_m, lambda_g), one panel per (lambda_orth, model/freq).
    Uses every NN run whose only differences from baseline are the lambdas."""
    plt = _plt()
    lam_vars = {"baseline", "lambda_m0", "lambda_g0", "lambda_orth0"} | {v for v in df.variant if v.startswith("grid_")}
    d = df[df.variant.isin(lam_vars)]
    if d.empty:
        return None
    cells = _cells(d)
    los = sorted(d.lambda_orth.unique())
    fig, axes = plt.subplots(len(cells), len(los), figsize=(3.6 * len(los), 3.0 * len(cells)), squeeze=False)
    for i, (m, f) in enumerate(cells):
        for j, lo in enumerate(los):
            ax = axes[i, j]
            g = d[(d.model == m) & (d.freq == f) & (d.lambda_orth == lo)]
            lm, lg = sorted(d.lambda_moment.unique()), sorted(d.lambda_g.unique())
            Z = np.full((len(lm), len(lg)), np.nan)
            for (a, b), gg in g.groupby(["lambda_moment", "lambda_g"]):
                Z[lm.index(a), lg.index(b)] = gg.val_nll.mean()
            im = ax.imshow(Z, origin="lower", cmap="viridis_r", aspect="auto")
            for (a, b), z in np.ndenumerate(Z):
                if np.isfinite(z):
                    ax.text(b, a, f"{z:.3f}", ha="center", va="center", fontsize=7, color="w")
            ax.set_xticks(range(len(lg)), [f"{x:g}" for x in lg])
            ax.set_yticks(range(len(lm)), [f"{x:g}" for x in lm])
            ax.set_xlabel("lambda_g")
            ax.set_ylabel("lambda_m")
            ax.set_title(f"{m}/{f}, lambda_orth={lo:g}", fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("P2: validation NLL (seed mean)")
    fig.tight_layout()
    p = os.path.join(out, "P2_lambda_grid.png")
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return p


def plot_p3(rows, out, dpi=120, variants=None):
    """Learning curves: train loss and val NLL by epoch, one line per seed, per variant."""
    plt = _plt()
    rows = [r for r in rows if r["history"]["train_loss"]]
    vs = list(dict.fromkeys(r["variant"] for r in rows if variants is None or r["variant"] in variants))
    if not vs:
        return None
    cells = list(dict.fromkeys((r["model"], r["freq"]) for r in rows))
    fig, axes = plt.subplots(len(cells), 2, figsize=(10, 3 * len(cells)), squeeze=False)
    cmap = plt.get_cmap("tab20")
    for i, c in enumerate(cells):
        for vi, v in enumerate(vs):
            first = True
            for r in rows:
                if (r["model"], r["freq"]) != c or r["variant"] != v:
                    continue
                h = r["history"]
                kw = dict(color=cmap(vi % 20), lw=0.9, alpha=0.8, label=v if first else None)
                axes[i, 0].plot(range(1, len(h["train_loss"]) + 1), h["train_loss"], **kw)
                axes[i, 1].plot(range(len(h["val_nll"])), h["val_nll"], **kw)
                first = False
        axes[i, 0].set_title(f"{c[0]}/{c[1]}: train loss", fontsize=9)
        axes[i, 1].set_title(f"{c[0]}/{c[1]}: val NLL (epoch 0 = stage 1)", fontsize=9)
        for ax in axes[i]:
            ax.set_xlabel("epoch")
    axes[0, 1].legend(fontsize=6, ncol=2)
    fig.suptitle("P3: learning curves (one line per seed)")
    fig.tight_layout()
    p = os.path.join(out, "P3_learning_curves.png")
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return p


def plot_p4(rows, out, dpi=120, hi=0):
    """Paired DM-statistic matrix per model/freq; * marks Holm-adjusted p < 0.05."""
    plt = _plt()
    cells = list(dict.fromkeys((r["model"], r["freq"]) for r in rows))
    mats = [(c, dm_matrix(rows, out, *c, hi=hi)) for c in cells]
    mats = [(c, m) for c, m in mats if len(m[0]) >= 2]
    if not mats:
        return None
    n = len(mats[0][1][0])
    fig, axes = plt.subplots(1, len(mats), figsize=(max(4, 0.45 * n + 2) * len(mats), max(3.5, 0.45 * n + 1.5)),
                             squeeze=False)
    k = next(r[f"h{hi}"] for r in rows)
    for ax, ((m, f), (names, S, P)) in zip(axes[0], mats):
        lim = np.nanmax(np.abs(S)) if np.isfinite(S).any() else 1
        im = ax.imshow(S, cmap="RdBu_r", vmin=-lim, vmax=lim)
        for (i, j), s in np.ndenumerate(S):
            if np.isfinite(s):
                ax.text(j, i, f"{s:.1f}{'*' if P[i, j] < 0.05 else ''}", ha="center", va="center", fontsize=6)
        ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=7)
        ax.set_yticks(range(len(names)), names, fontsize=7)
        ax.set_title(f"{m}/{f}, QLIKE k={k}", fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8, label="DM stat (row - col); <0 row better")
    fig.suptitle("P4: paired DM (HAC), * = Holm p<0.05")
    fig.tight_layout()
    p = os.path.join(out, f"P4_dm_h{hi}.png")
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return p


P5_PARAMS = ("omega", "beta", "gamma", "persistence", "phi", "tau1", "tau2", "sigma_u")


def plot_p5(df, out, dpi=120):
    """Per economic parameter: (variant - baseline) final-parameter shift and the stage 2 -
    stage 1 change within the variant, both mean with 95% seed CIs."""
    plt = _plt()
    ps = [p for p in P5_PARAMS if f"p_{p}" in df]
    cells = _cells(df)
    variants = list(dict.fromkeys(df["variant"]))
    fig, axes = plt.subplots(len(ps), 2, figsize=(max(8, 0.5 * len(variants) * len(cells) * 2), 2.0 * len(ps)),
                             squeeze=False)
    w = 0.8 / max(len(cells), 1)
    x = np.arange(len(variants))
    for pi, p in enumerate(ps):
        for ci, (m, f) in enumerate(cells):
            g = df[(df.model == m) & (df.freq == f)]
            b = g[g.variant == "baseline"][f"p_{p}"].mean()
            s_vb = [tci(g[g.variant == v][f"p_{p}"] - b) for v in variants]
            s_21 = [tci(g[g.variant == v][f"p_{p}"] - g[g.variant == v][f"s1_{p}"]) for v in variants]
            for ax, st in ((axes[pi, 0], s_vb), (axes[pi, 1], s_21)):
                ax.errorbar(x + ci * w, [s[0] for s in st], yerr=[s[2] for s in st], fmt="o", ms=3,
                            capsize=2, label=f"{m}/{f}")
        for ax in axes[pi]:
            ax.axhline(0, color="k", lw=0.5)
            ax.set_ylabel(p, fontsize=8)
            ax.set_xticks(x + w * (len(cells) - 1) / 2)
            ax.set_xticklabels(variants if pi == len(ps) - 1 else [], rotation=50, ha="right", fontsize=7)
    axes[0, 0].set_title("variant - baseline (final params)", fontsize=9)
    axes[0, 1].set_title("stage 2 - stage 1 within variant", fontsize=9)
    axes[0, 0].legend(fontsize=6)
    fig.suptitle("P5: parameter shifts (mean, 95% seed CI)")
    fig.tight_layout()
    p = os.path.join(out, "P5_param_shifts.png")
    fig.savefig(p, dpi=dpi)
    plt.close(fig)
    return p


def make_plots(df, rows, out, dpi=120):
    paths = [plot_p1(df, out, dpi), plot_p2(df, out, dpi), plot_p3(rows, out, dpi),
             plot_p5(df, out, dpi)]
    n_h = sum(1 for c in df.columns if c.startswith("qlike_h"))
    paths += [plot_p4(rows, out, dpi, hi=i) for i in range(n_h)]
    return [p for p in paths if p]
