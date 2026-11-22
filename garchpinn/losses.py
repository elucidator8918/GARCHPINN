"""Training loss (doc Section 7, eq. pinn-loss) and moment residuals (eq. moments)."""
import torch


def _mmean(x, m):
    return (x * m).sum() / m.sum().clamp_min(1.0)


def moment_residuals(z, u, sigma_u, mask=None, lags=5, standardize=False):
    """R_moment = ||m_bar||^2 for the conditions of eq. (moments).

    standardize=True divides each mean by its (detached) standard error, i.e. GMM with a
    diagonal weight W = diag(n / var(m_j)) divided by n: R = (1/n) sum_j (m_bar_j / se_j)^2.
    Without it, the minibatch noise floor sum_j var(m_j)/n (about 0.1-0.2 for a 1000-obs
    TBPTT chunk, dominated by the e_t e_{t-l} products, var ~ 4) dwarfs the signal and the
    penalty pulls the econ parameters off the QMLE (the cause of stage-2 regressions).

    z, u: (L,) or (B, L). u is standardised by a constant (sigma_u detached): standardising
    by the live parameter would let the penalty shrink by inflating sigma_u. Lagged products use only pairs
    (t, t-l) inside the same row with both positions unmasked; lags never wrap across rows.
    """
    if z.dim() == 1:
        z, u = z[None], u[None]
        mask = mask[None] if mask is not None else None
    m = torch.ones_like(z) if mask is None else mask.to(z.dtype)
    u = u / torch.as_tensor(sigma_u).detach()
    e = z * z - 1
    terms = [(z, m), (e, m), (u, m), (u * z, m), (u * e, m)]
    L = z.shape[1]
    for l in range(1, lags + 1):
        if l >= L:
            break
        ml = m[:, l:] * m[:, :-l]
        terms += [(e[:, l:] * e[:, :-l], ml), (u[:, l:] * u[:, :-l], ml), (u[:, l:] * z[:, :-l], ml)]
    mom = torch.stack([_mmean(v, w) for v, w in terms])
    if not standardize:
        return mom.pow(2).sum()
    with torch.no_grad():  # standard error of each mean (constant weight, as in two-step GMM)
        se2 = torch.stack([_mmean(v * v, w) / w.sum().clamp_min(1.0) for v, w in terms]).clamp_min(1e-12)
    return (mom.pow(2) / se2).sum() / m.sum().clamp_min(1.0)


def _lag_regressors(model, out, log_x):
    """Detached F_{t-1} regressors of the GARCH equation that g[:, t] must not span (beyond
    the constant): log h_{t-1} always; log x_{t-1} (GARCH) or z_{t-1}, z_{t-1}^2 - 1, u_{t-1}
    (EGARCH). Each is (B, L-1), aligned with g[:, 1:]."""
    two = lambda v: v if v.dim() > 1 else v[None]
    lh, z, u = two(out["log_h"]), two(out["z"]), two(out["u"])
    regs = [lh[:, :-1]]
    if getattr(model, "variant", None) == "egarch":
        regs += [z[:, :-1], z[:, :-1] ** 2 - 1, u[:, :-1]]
    elif log_x is not None:
        regs.append(two(log_x)[:, :-1])
    return [r.detach() for r in regs]


def orth_penalty(model, out, mask, log_x=None):
    """P_orth = mean(g)^2 + sum_k cov(g_t, q_{k,t-1} / sd(q_k))^2 over the GARCH regressors
    q_k (doc eq. pinn-orth). Regressors are standardised (detached) so the terms are in the
    units of g and comparable across k."""
    g = out["g"] if out["g"].dim() > 1 else out["g"][None]
    m = torch.ones_like(g) if mask is None else mask.to(g.dtype)
    m = m if m.dim() > 1 else m[None]
    gm = _mmean(g, m)
    ml = m[:, 1:] * m[:, :-1]
    po = gm ** 2
    for q in _lag_regressors(model, out, log_x):
        qc = q - _mmean(q, ml)
        sd = _mmean(qc * qc, ml).sqrt().clamp_min(1e-8)
        po = po + _mmean((g[:, 1:] - gm) * qc / sd, ml) ** 2
    return po


def total_loss(model, out, mask, cfg, log_x=None):
    """nll_r + nll_x + lambda_moment R_moment + lambda_g mean(g^2) + lambda_orth P_orth.
    (lambda_w via Adam weight decay.) log_x: the (B, L) log realized measure of the chunk,
    needed for the GARCH orthogonality term cov(g_t, log x_{t-1})."""
    nll_r, nll_x = model.nll_parts(out, mask)
    loss = nll_r + nll_x if getattr(cfg, "likelihood", "joint") == "joint" else nll_r
    parts = {"nll_r": nll_r, "nll_x": nll_x}
    if cfg.lambda_moment > 0:
        rm = moment_residuals(out["z"], out["u"], model.sigma_u, mask, cfg.moment_lags,
                              standardize=getattr(cfg, "moment_standardize", True))
        loss = loss + cfg.lambda_moment * rm
        parts["moment"] = rm
    if cfg.lambda_g > 0 and model.nn_active:
        m = torch.ones_like(out["g"]) if mask is None else mask.to(out["g"].dtype)
        pg = _mmean(out["g"] ** 2, m)
        loss = loss + cfg.lambda_g * pg
        parts["g2"] = pg
    lam_o = getattr(cfg, "lambda_orth", 0.0)
    if lam_o > 0 and model.nn_active:
        po = orth_penalty(model, out, mask, log_x)
        loss = loss + lam_o * po
        parts["orth"] = po
    # detached tensors; caller converts to float once per epoch (avoids per-batch syncs)
    return loss, {k: v.detach() for k, v in parts.items()}
