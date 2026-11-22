"""Two-stage fitting (doc Algorithm 2): stage 1 QMLE with L-BFGS, stage 2 PINN with Adam + TBPTT."""
import copy
import math

import torch

from .data import streams
from .losses import total_loss


def _stream_mask(n_streams, L, burn_in, like):
    m = torch.ones(n_streams, L, dtype=like.dtype, device=like.device)
    # every stream, including stream 0, starts from log h_1 = log var(r_1..r_22), which uses
    # the same returns: mask the burn-in so the objective conditions on a pre-sample h_1
    m[:, : min(burn_in, L)] = 0.0
    return m


def _segments(ds, sl, n):
    return streams(ds.r[sl], n), streams(ds.log_x[sl], n), streams(ds.exo[sl], n)


def stage1_qmle(model, ds, cfg):
    """Pure Realized (E)GARCH QMLE over econ parameters only; returns per-obs joint NLL."""
    model.nn_active = False
    with torch.no_grad():  # centre of the log h guard: training log variance
        model.lh_center.fill_(float(torch.log(ds.r[ds.split("train")].var().clamp_min(1e-12))))
    r, lx, ex = _segments(ds, ds.split("train"), cfg.stage1_streams)
    mask = _stream_mask(r.shape[0], r.shape[1], cfg.stage1_burn_in, r)
    params = model.econ_parameters()
    opt = torch.optim.LBFGS(params, lr=1.0, max_iter=cfg.stage1_iters, history_size=50,
                            tolerance_grad=1e-9, tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        out, _ = model(r, lx, ex)
        a, b = model.nll_parts(out, mask)
        loss = a + b if getattr(cfg, "likelihood", "joint") == "joint" else a
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        out, _ = model(r, lx, ex)
        a, b = model.nll_parts(out, mask)
    return {"stage1_nll": float(a + b), "stage1_nll_r": float(a), "stage1_nll_x": float(b)}


@torch.no_grad()
def val_nll(model, ds, partial=False):
    """Filter train+val from scratch; joint (or, if partial, return-only) NLL per obs over val."""
    sl = ds.split("fit")
    out, _ = model(ds.r[sl], ds.log_x[sl], ds.exo[sl])
    mask = torch.zeros_like(out["z"])
    mask[ds.n_train:] = 1.0
    a, b = model.nll_parts(out, mask)
    return a if partial else a + b


def stage2_pinn(model, ds, cfg, log=None):
    hist = {"train_loss": [], "val_nll": [], "parts": []}
    if not cfg.use_nn or not model.use_nn:
        return hist
    model.nn_active = True
    # econ params start at the QMLE optimum: Adam's sign-like steps of size ~lr would kick
    # them off it before g has learned anything (val NLL rose for the first ~5 epochs and
    # early stopping then discarded stage 2). So: freeze them for econ_warmup epochs, then
    # move them with a smaller lr; cosine-decay both groups.
    warm = int(getattr(cfg, "econ_warmup", 0))
    econ_lr = cfg.lr * getattr(cfg, "econ_lr_mult", 1.0)
    opt = torch.optim.Adam([
        {"params": model.econ_parameters(), "weight_decay": 0.0, "lr": 0.0 if warm > 0 else econ_lr},
        {"params": model.nn_parameters(), "weight_decay": cfg.weight_decay, "lr": cfg.lr},
    ], lr=cfg.lr)
    base_lr = [econ_lr, cfg.lr]
    cosine = getattr(cfg, "lr_schedule", "cosine") == "cosine"
    r, lx, ex = _segments(ds, ds.split("train"), cfg.streams)
    S, L = r.shape
    full_mask = _stream_mask(S, L, cfg.stage1_burn_in, r)
    partial = getattr(cfg, "likelihood", "joint") == "partial"
    carry = getattr(cfg, "carry_state", True)
    best = val_nll(model, ds, partial)
    best_state, best_epoch, bad = copy.deepcopy(model.state_dict()), 0, 0
    hist["val_nll"].append(float(best))
    for epoch in range(1, cfg.epochs + 1):
        fac = 0.5 * (1 + math.cos(math.pi * (epoch - 1) / max(cfg.epochs, 1))) if cosine else 1.0
        for grp, lr0 in zip(opt.param_groups, base_lr):
            grp["lr"] = lr0 * fac
        if epoch <= warm:
            opt.param_groups[0]["lr"] = 0.0
        model.train()
        state = None
        tot = torch.zeros((), dtype=r.dtype, device=r.device)
        sums, nb = {}, 0
        for a in range(0, L, cfg.chunk_len):
            b = min(a + cfg.chunk_len, L)
            m = full_mask[:, a:b]
            if m.sum() == 0:  # chunk fully in burn-in: advance state only
                with torch.no_grad():
                    _, state = model(r[:, a:b], lx[:, a:b], ex[:, a:b], state)
                continue
            if not carry:
                state = None
            out, state = model(r[:, a:b], lx[:, a:b], ex[:, a:b], state)
            loss, parts = total_loss(model, out, m, cfg, lx[:, a:b])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
            state = {k: v.detach() for k, v in state.items()}  # truncated BPTT
            tot = tot + loss.detach()
            for k, v in parts.items():
                sums[k] = sums.get(k, 0.0) + v
            nb += 1
        model.eval()
        v = float(val_nll(model, ds, partial))  # one sync per epoch
        hist["train_loss"].append(float(tot) / max(nb, 1))
        hist["parts"].append({k: float(s) / max(nb, 1) for k, s in sums.items()})
        hist["val_nll"].append(v)
        if log is not None:
            log({"epoch": epoch, "train_loss": hist["train_loss"][-1], "val_nll": v, **hist["parts"][-1]})
        if v < float(best) - 1e-8:
            best, best_state, best_epoch, bad = v, copy.deepcopy(model.state_dict()), epoch, 0
        elif epoch > warm:  # patience counts only after the warm-up
            bad += 1
            if bad >= cfg.patience:
                break
    model.load_state_dict(best_state)
    hist["best_epoch"], hist["best_val_nll"] = best_epoch, float(best)
    return hist


def fit(model, ds, cfg, log=None, on_stage1=None):
    """Stage 1 then stage 2. on_stage1(model), if given, is called after stage 1 (NN off),
    e.g. to keep a copy of the pure econometric model."""
    hist = stage1_qmle(model, ds, cfg)
    model.nn_active = False
    hist["stage1_val_nll"] = float(val_nll(model, ds))
    if on_stage1 is not None:
        on_stage1(model)
    if log is not None:
        log({k: v for k, v in hist.items()})
    hist.update(stage2_pinn(model, ds, cfg, log))
    model.nn_active = bool(cfg.use_nn and model.use_nn)
    finalize(model, ds)
    return hist


@torch.no_grad()
def finalize(model, ds):
    """Training-split quantities used by the forecasts: mean of g (steps >= 2 of the closed
    form) and residuals (z_hat, u_hat) for the empirical MGF."""
    sl = ds.split("train")
    out, _ = model(ds.r[sl], ds.log_x[sl], ds.exo[sl])
    model.g_bar.fill_(float(out["g"].mean()))
    model.set_empirical_mgf(out["z"], out["u"])
    return out
