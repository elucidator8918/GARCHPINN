"""Fit, evaluate and forecast GARCHPINN / EGARCHPINN on FTSE data.

    python main.py                                  # all 4 configurations
    python main.py --model egarch --freq daily
    python main.py --no-nn                          # pure Realized (E)GARCH (QMLE only)
    python main.py --device cuda --parallel-configs # 4 configs at once, one process per GPU
"""
import argparse
import copy
import json
import os

import torch

torch.set_default_dtype(torch.float64)

from garchpinn import train as T  # noqa: E402
from garchpinn.config import FREQS, MODELS, Config  # noqa: E402
from garchpinn.evaluate import evaluate  # noqa: E402
from garchpinn.data import build_dataset, future_cal_angles, get_series  # noqa: E402
from garchpinn.models import build_model  # noqa: E402
from garchpinn.plots import plot_fit  # noqa: E402
from garchpinn.utils import (_jsonable, assign_devices, make_generator, pin_device,  # noqa: E402
                             resolve_device, save_model, set_seeds)


def fit_two_stage(model, ds, cfg, log=None):
    """Run train.fit and return (history, pure_model), pure_model being the stage-1 copy."""
    holder = {}
    hist = T.fit(model, ds, cfg, log=log, on_stage1=lambda m: holder.setdefault("m", copy.deepcopy(m)))
    pure = holder["m"]
    pure.nn_active = False
    pure.g_bar.zero_()
    T.finalize(pure, ds)
    return hist, pure


def run_evaluate(model, pure, ds, cfg, return_series=False):
    return evaluate(model, ds, cfg, pure_model=pure, return_series=return_series)


def _step(name, fn, store):
    """Run one diagnostic; a failure is reported and recorded, never fatal."""
    try:
        return fn()
    except Exception as e:  # noqa: BLE001
        print(f"diagnostic {name} failed: {type(e).__name__}: {e}")
        store.setdefault("errors", {})[name] = f"{type(e).__name__}: {e}"
        return None


def run_diagnostics(model, pure, ds, cfg, df, out, state, fc, metrics, series, od, fast=False,
                    recovery=False, rolling=False):
    """Doc Section figs: F1-F7, R1-R5, I1-I5, E2-E6. Returns the numeric summary."""
    import numpy as np
    from garchpinn import diagnostics as D
    from garchpinn import plots as P
    res = {}
    unit = "days" if cfg.freq == "daily" else "bars"
    with torch.no_grad():
        pout, pstate = pure(ds.r, ds.log_x, ds.exo)
    H = cfg.forecast_horizon
    # ---- model output
    _step("F1", lambda: P.plot_F1(ds, out, od, pure_out=pout), res)
    _step("F2", lambda: P.plot_F2(model, out, ds, od), res)
    irf = _step("F3", lambda: D.impulse_response(model, H, state=state, n_paths=cfg.irf_paths, seed=cfg.seed), res)
    if irf is not None:
        _step("F3", lambda: P.plot_F3(irf, od, unit), res)
        res["impulse"] = {"pi": irf["pi"], "half_life": irf["half_life"]}
    q = _step("F4q", lambda: D.forecast_quantiles(model, state, H, 2000, cfg.seed), res)
    _step("F4", lambda: P.plot_F4(fc, od, q), res)
    Ks = [k for k in ((1, 5, 10, 22) if cfg.freq == "daily" else (1, 12, 51, 102)) if k <= H]
    te = np.arange(ds.split("test").start, ds.T)
    ts = _step("F5", lambda: D.term_structure(model, out["log_h"], te, Ks), res)
    if ts is not None:
        _step("F5", lambda: P.plot_F5(ts, od), res)
    if cfg.freq == "5min":
        _step("F6", lambda: P.plot_F6(df, cfg, od), res)
    if model.nn_active:
        def f7():
            X, names = D.nn_inputs(model, ds, out)
            pdr = D.partial_dependence(model, X, seed=cfg.seed)
            g = out["g"]
            res["g"] = {"mean": float(g.mean()), "sd": float(g.std())}
            return P.plot_F7(ds, g, X, names, pdr, cfg.lambda_g, od)
        _step("F7", f7, res)
    # ---- residuals
    z, u = D.residuals(out, model)
    lbl = (10, 22)
    rs = {sp: D.residual_stats(z[sl], u[sl], lbl) for sp, sl in (("train", ds.split("train")), ("test", ds.split("test")))}
    res["residuals"] = rs
    _step("R1", lambda: P.plot_R1(z, u, ds, rs, od), res)
    _step("R2", lambda: P.plot_R2(z, u, ds, rs, 22 if cfg.freq == "daily" else 390, od), res)
    _step("R3", lambda: P.plot_R3(D.moment_paths(z, u, lags=cfg.moment_lags), ds, od), res)
    _step("R4", lambda: P.plot_R4(z, ds, od), res)
    _step("R5", lambda: P.plot_R5(z, u, ds, od), res)
    # ---- inference (stage-1 model)
    se = _step("I2", lambda: D.standard_errors(pure, ds, cfg.stage1_burn_in, cfg.diag_max_obs), res)
    if se is not None:
        res["standard_errors"] = {k: se[k] for k in ("names", "estimate", "se_hessian", "se_sandwich", "ratio", "n")}
        _step("I2", lambda: P.plot_I2(se, od), res)
    if not fast:
        prof = _step("I1", lambda: D.profile_likelihood(
            pure, ds, se, n_grid=cfg.profile_grid, iters=cfg.profile_iters, burn_in=cfg.stage1_burn_in,
            max_obs=cfg.profile_max_obs), res)
        if prof is not None:
            res["profile"] = prof
            _step("I1", lambda: P.plot_I1(prof, od), res)
    if recovery:
        rec = _step("I3", lambda: D.parameter_recovery(cfg.model, pure.params(), cfg=cfg, seed=cfg.seed), res)
        if rec is not None:
            res["recovery"] = rec
            _step("I3", lambda: P.plot_I3(rec, od), res)
    cloud = None
    if rolling:
        roll = _step("I4", lambda: D.rolling_params(pure, ds, cfg), res)
        if roll is not None:
            res["rolling"] = roll
            _step("I4", lambda: P.plot_I4(roll, od), res)
            cloud = [(r["params"]["beta"], r["params"]["gamma"] * (r["params"]["phi"] if cfg.model == "garch" else 1))
                     for r in roll]
    st = D.stationarity_point(pure, se)
    pp = model.pinn_persistence(out, ds.exo)
    pinn_pt = None
    if pp is not None:
        pinn_pt = ((pp - st["y"], st["y"]) if cfg.model == "garch" else (pp, st["y"]))
    _step("I5", lambda: P.plot_I5(st, od, pinn_pt, cloud), res)
    res["stationarity"] = {k: st[k] for k in ("x", "y") if k in st}
    # ---- forecast evaluation
    cum = {hk: D.cumulative_loss_diff(s) for hk, s in series.items()}
    _step("E2", lambda: P.plot_E2(cum, metrics, od), res)
    mz = {hk: D.mincer_zarnowitz(s["target"], s["forecast"]["model"], int(hk[2:])) for hk, s in series.items()}
    res["mincer_zarnowitz"] = mz
    _step("E3", lambda: P.plot_E3(series, mz, od), res)
    if not fast:
        names = [n for n in series[next(iter(series))]["loss"] if n != "garch11_raw"]
        mcs = {hk: D.model_confidence_set({n: s["loss"][n]["qlike"] for n in names}, cfg.mcs_alpha,
                                          cfg.mcs_boot, seed=cfg.seed) for hk, s in series.items()}
        res["mcs"] = mcs
        _step("E4", lambda: P.plot_E4(mcs, od), res)
    cal = {hk: D.calibration_deciles(s["target"], s["forecast"]["model"]) for hk, s in series.items()}
    res["calibration"] = cal
    _step("E5", lambda: P.plot_E5(cal, od), res)
    sl = ds.split("test")
    rte = ds.r[sl].cpu().numpy()
    var = D.var_backtest(rte, float(model.mu), torch.exp(out["log_h"][sl]).cpu().numpy())
    res["var_backtest"] = {a: {k: v for k, v in d.items() if k not in ("q", "hits_mask")} for a, d in var.items()}
    _step("E6", lambda: P.plot_E6(rte, var, od), res)
    return res


def run(cfg: Config, cache):
    set_seeds(cfg.seed)
    if cfg.freq not in cache:
        cache[cfg.freq] = get_series(cfg)
    dev = resolve_device(cfg.device, cfg)
    cfg.device = dev
    pin_device(dev)
    ds = build_dataset(cache[cfg.freq], cfg, device=dev)
    n_exo = ds.exo.shape[-1]
    print(f"\n=== {cfg.name}: T={ds.T} (train {ds.n_train}, val {ds.n_val}) on {dev} ===")
    model = build_model(cfg, n_exo).to(dev)

    log = None
    if cfg.use_wandb:
        import wandb
        wandb.init(project="garchpinn-ftse", name=cfg.name, config=cfg.to_dict(), reinit=True)
        log = wandb.log

    history, pure = fit_two_stage(model, ds, cfg, log)
    model.eval()
    print("params:", _jsonable(model.params()))

    metrics, series = run_evaluate(model, pure, ds, cfg, return_series=True)

    with torch.no_grad():  # filter whole sample, forecast from last state
        out, state = model(ds.r, ds.log_x, ds.exo)
        H = cfg.forecast_horizon
        W = max(cfg.har_windows)
        t = lambda a: torch.as_tensor(a, dtype=ds.r.dtype, device=ds.r.device)
        exo_ctx = None
        if ds.exo_mu is not None:
            exo_ctx = {"lx_hist": ds.log_x[-W:], "windows": cfg.har_windows, "mu": t(ds.exo_mu),
                       "sd": t(ds.exo_sd), "cal_ang": t(future_cal_angles(cache[cfg.freq], cfg, H + 1))}
        gb = model.g_bar if model.nn_active else 0.0
        mg = cfg.forecast_mgf if model.emp_resid is not None else "gaussian"
        alt, suf = ("empirical", "_emp") if mg == "gaussian" else ("gaussian", "_gauss")
        fc = {"analytic": model.forecast(out["log_h_next"], H, g_bar=gb, mgf=mg),
              "mc": model.simulate_forecast(state, H, cfg.n_paths, make_generator(cfg.seed, ds.r.device),
                                            exo_ctx=exo_ctx)}
        if model.emp_resid is not None:
            fc["analytic" + suf] = model.forecast(out["log_h_next"], H, g_bar=gb, mgf=alt)
        pout, _ = pure(ds.r, ds.log_x, ds.exo)
        pmg = cfg.forecast_mgf if pure.emp_resid is not None else "gaussian"
        fc["pure"] = pure.forecast(pout["log_h_next"], H, mgf=pmg)
        pp = model.pinn_persistence(out, ds.exo)

    od = os.path.join(cfg.out_dir, cfg.name)
    os.makedirs(od, exist_ok=True)
    save_model(model, os.path.join(od, "model"), cfg)
    save_model(pure, os.path.join(od, "pure_model"), cfg)
    plot_fit(cfg.name, ds, out, history, fc, metrics, od)
    params = model.params()
    if pp is not None:
        params["pinn_persistence"] = pp
    summary = {"name": cfg.name, "config": cfg.to_dict(), "params": params,
               "pure_params": pure.params(), "history": history, "metrics": metrics,
               "forecast": {k: {kk: vv for kk, vv in v.items()} for k, v in fc.items()}}
    if cfg.diagnostics:
        summary["diagnostics"] = run_diagnostics(
            model, pure, ds, cfg, cache[cfg.freq], out, state, fc, metrics, series, od,
            fast=cfg.diag_fast, recovery=cfg.diag_recovery, rolling=cfg.diag_rolling)
    with open(os.path.join(od, "summary.json"), "w") as f:
        json.dump(_jsonable(summary), f, indent=2, default=str)
    if log is not None:
        import wandb
        wandb.finish()
    return summary


def _run_worker(cfg):
    import torch as _t
    _t.set_default_dtype(_t.float64)
    return cfg.name, cfg.device, run(cfg, {})["metrics"]


def run_parallel(cfgs, device="auto", gpus=None):
    """One process per configuration (spawn); with CUDA, configs go round-robin over GPUs."""
    import concurrent.futures as cf
    import multiprocessing as mp
    devs = assign_devices(len(cfgs), device, gpus, cfgs[0])
    nthreads = max(1, (os.cpu_count() or 1) // len(cfgs))
    for cfg, d in zip(cfgs, devs):
        cfg.device = d
        print(f"config {cfg.name} -> {d}")
    res = {}
    with cf.ProcessPoolExecutor(len(cfgs), mp_context=mp.get_context("spawn"),
                                initializer=torch.set_num_threads, initargs=(nthreads,)) as ex:
        for name, d, met in ex.map(_run_worker, cfgs):
            res[name] = met
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=[*MODELS, "all"], default="all")
    ap.add_argument("--freq", choices=[*FREQS, "all"], default="all")
    ap.add_argument("--data", default=None, help="path to the FTSE 5-min CSV")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--no-nn", action="store_true")
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--diagnostics", action=argparse.BooleanOptionalAction, default=True,
                    help="doc figures F1-F7, R1-R5, I1-I5, E2-E6 (default on)")
    ap.add_argument("--fast", action="store_true", help="skip the profile likelihood and the MCS")
    ap.add_argument("--recovery", action="store_true", help="I3 parameter recovery by simulation (slow)")
    ap.add_argument("--rolling", action="store_true", help="I4 rolling-window parameters (slow)")
    ap.add_argument("--device", default="auto",
                    help="auto (CUDA only for wide workloads), cpu, cuda, cuda:K; never MPS")
    ap.add_argument("--gpus", default=None, help="GPU ids for cuda, e.g. 0,1,2 (default: all visible)")
    ap.add_argument("--parallel-configs", action="store_true",
                    help="run the configurations concurrently, one process per GPU (or per CPU slot)")
    a = ap.parse_args()

    models = MODELS if a.model == "all" else (a.model,)
    freqs = FREQS if a.freq == "all" else (a.freq,)
    cache, all_res, cfgs = {}, {}, []
    for fq in freqs:
        for m in models:
            kw = dict(model=m, freq=fq, use_nn=not a.no_nn, use_wandb=a.wandb, out_dir=a.out,
                      diagnostics=a.diagnostics, diag_fast=a.fast, diag_recovery=a.recovery,
                      diag_rolling=a.rolling)
            if a.data:
                kw["data_path"] = a.data
            cfg = Config(**kw)
            if a.epochs is not None:
                cfg.epochs = a.epochs
            # plain "cuda" -> the first of --gpus (run_parallel reassigns round-robin)
            cfg.device = resolve_device(a.device, cfg, a.gpus) if a.device == "cuda" else a.device
            cfgs.append(cfg)
    if a.parallel_configs and len(cfgs) > 1:
        all_res = run_parallel(cfgs, a.device, a.gpus)
    else:
        for cfg in cfgs:
            all_res[cfg.name] = run(cfg, cache)["metrics"]
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, "summary.json"), "w") as f:
        json.dump(_jsonable(all_res), f, indent=2, default=str)


if __name__ == "__main__":
    main()
