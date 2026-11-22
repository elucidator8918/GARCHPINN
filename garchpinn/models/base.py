"""Shared machinery for GARCHPINN / EGARCHPINN (reference doc Sections 3-7).

Return eq.       r_t = mu + sqrt(h_t) z_t
Measurement eq.  log x_t = xi + phi log h_t + d(z_t) + u_t,  d(z) = d1 z + d2 (z^2 - 1)
GARCH eq.        subclass-specific, plus the neural correction s*g(f_{t-1}) (F_{t-1}-measurable)
Loss             -(1/n) l(r, x) + lambda_m R_moment + lambda_g mean(g^2)  [+ weight decay]

Every model is written in the unified state-space form used for forecasting:
    log h_{t+1} = c + a log h_t + w_t + g_t,
    w_t = A1 z_t + A2 (z_t^2 - 1) + Bu u_t    (mean-zero shock built from z_t, u_t)
so the k-step mean and the log-normal (Jensen) corrections are shared code (Section 5).
"""
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

LOG2PI = math.log(2 * math.pi)
PERSIST_MAX = 0.9999


def _logit(p):
    return math.log(p / (1 - p))


def psi(c, a1, a2, b, sigma_u):
    """log E exp(c [a1 z + a2 (z^2-1) + b u]) for z ~ N(0,1), u ~ N(0, sigma_u^2) independent.
    Valid when 1 - 2 c a2 > 0 (reference doc eq. mgf)."""
    q = 1 - 2 * c * a2
    if torch.any(torch.as_tensor(q) <= 0):
        raise ValueError("moment generating function undefined: 1 - 2 c a2 <= 0")
    return -0.5 * torch.log(q) - c * a2 + (c * a1) ** 2 / (2 * q) + 0.5 * (c * b * sigma_u) ** 2


class RealizedPINN(nn.Module):
    variant = None  # "garch" | "egarch"

    def __init__(self, n_exo=0, hidden_dim=32, use_nn=True, nn_state_inputs=True,
                 nn_scale=0.25, log_h_bound=15.0, overnight=False):
        super().__init__()
        # overnight=True: the last two exo columns are (on_t, r_on_t), read by the likelihood
        # (doc Section 1, overnight return) and never fed to the network
        self.overnight = bool(overnight)
        self.n_exo_nn = n_exo - 2 if self.overnight else n_exo
        if self.overnight and self.n_exo_nn < 0:
            raise ValueError("overnight=True needs the (on, r_on) exo columns")
        P = lambda v: nn.Parameter(torch.tensor(float(v)))
        # typical magnitudes from HHS (2012) / HH (2016) as initial values (doc Section 6)
        self.mu = P(0.0)
        self.omega = P(0.0)
        self.gamma = P(0.4)
        self.xi = P(0.0)
        self.phi = P(1.0)
        self.d1 = P(-0.07)
        self.d2 = P(0.07)
        self.log_sigma_u = P(math.log(0.4))
        self.persist_raw = P(_logit(0.97 / PERSIST_MAX))
        if self.overnight:
            self.kappa = P(0.0)  # Var(r_on_t | F_{t-1}) = exp(kappa) h_t
        self.log_h_bound = log_h_bound
        # centre of the log h guard (set to the training log variance before stage 1) and
        # training mean of the NN correction (used for forecast steps >= 2)
        self.register_buffer("lh_center", torch.tensor(0.0))
        self.register_buffer("g_bar", torch.tensor(0.0))
        self.emp_resid = None  # (z, u) training residuals for the empirical MGF

        self.use_nn = use_nn
        self.nn_active = use_nn  # switched off during stage-1 QMLE
        self.nn_state_inputs = nn_state_inputs
        self.nn_scale = nn_scale
        if use_nn:
            n_in = self.n_exo_nn + (2 if nn_state_inputs else 0)
            self.net = nn.Sequential(nn.Linear(n_in, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, 1, bias=False))
            # no output bias: a free constant would duplicate omega (identification)
            nn.init.zeros_(self.net[-1].weight)  # start exactly at the econometric model

    # ------------------------------------------------------------ parameters
    @property
    def persistence(self):
        return PERSIST_MAX * torch.sigmoid(self.persist_raw)

    @property
    def sigma_u(self):
        return self.log_sigma_u.exp()

    def meas_lev(self, z):
        return self.d1 * z + self.d2 * (z * z - 1)

    def econ_parameters(self):
        return [p for n, p in self.named_parameters() if not n.startswith("net.") and p.requires_grad]

    lev_names = ("d1", "d2")  # leverage tau(z) parameters; EGARCH overrides

    @torch.no_grad()
    def freeze_leverage(self):
        """Ablation tau == 0: zero the leverage parameters and exclude them from training."""
        for n in self.lev_names:
            getattr(self, n).zero_()
            getattr(self, n).requires_grad_(False)
        return self

    def nn_parameters(self):
        return list(self.net.parameters()) if self.use_nn else []

    # ---- subclass interface ----
    def garch_step(self, lh, lx, z, u):
        """log h_{t+1} (without NN correction) from time-t quantities."""
        raise NotImplementedError

    def garch_coefs(self):
        """(A, Bh, Cx, Kz, Ke, Ku) with garch_step = A + Bh lh + Cx lx + Kz z + Ke (z^2-1) + Ku u."""
        raise NotImplementedError

    step_uses_shock = True  # garch_step depends on (z, u) beyond log x (EGARCH); GARCH: False

    def state_space(self):
        """(c, a, A1, A2, Bu) of the unified form in the module docstring."""
        raise NotImplementedError

    def params(self):
        raise NotImplementedError

    # ------------------------------------------------------------ filter
    def correction(self, f, z, lh):
        if not self.nn_active:
            return torch.zeros_like(lh)
        f = f[..., : self.n_exo_nn]
        inp = torch.cat([f, z[..., None], lh[..., None]], -1) if self.nn_state_inputs else f
        return self.nn_scale * torch.tanh(self.net(inp).squeeze(-1))

    def _bound(self, lh):
        """Numerical guard, identity up to ~exp(-(B-|d|)) for d = lh - centre, |d| < B.
        lh_c + d - softplus(d - B) + softplus(-d - B): saturates linearly-softly only far
        outside the data range, so the linear state-space form (and forecast()) holds."""
        B = self.log_h_bound
        c = self.lh_center
        d = lh - c
        return c + d - F.softplus(d - B) + F.softplus(-d - B)

    def _check_guard(self, m):
        if torch.any((m - self.lh_center).abs() > self.log_h_bound - 5):
            warnings.warn("forecast log h within 5 of the guard bound: closed form inexact")

    def init_log_h(self, r, n=22):
        """log of the sample variance of r over the first n obs (doc Section 6), detached."""
        n = min(n, r.shape[-1])
        return torch.log(r[..., :n].var(-1, unbiased=False).clamp_min(1e-12)).detach()

    def _prep(self, r, log_x, exo, state):
        squeeze = r.dim() == 1
        if squeeze:
            r, log_x = r[None], log_x[None]
            exo = exo[None] if exo is not None else None
            if state is not None:
                state = {k: v[None] if v.dim() < (2 if k == "f" else 1) else v for k, v in state.items()}
        if exo is None:
            exo = r.new_zeros(r.shape[0], r.shape[1], 0)
        if state is None:
            lh = self.init_log_h(r)
            g0 = torch.zeros_like(lh)
        else:
            g0 = self.correction(state["f"], state["z"], state["lh"])
            lh = self._bound(self.garch_step(state["lh"], state["lx"], state["z"], state["u"]) + g0)
        return squeeze, r, log_x, exo, lh, g0

    def _finish(self, squeeze, r, log_x, exo, lh_all, g_all):
        """Vectorised post-processing: z, u from the filtered log h path, state and forecast."""
        z_all = (r - self.mu) * torch.exp(-0.5 * lh_all)
        u_all = log_x - self.xi - self.phi * lh_all - self.meas_lev(z_all)
        out = {"log_h": lh_all, "z": z_all, "u": u_all, "g": g_all}
        if self.overnight:
            out["on"], out["r_on"] = exo[..., -2], exo[..., -1]
        lh, z, u = lh_all[:, -1], z_all[:, -1], u_all[:, -1]
        new_state = {"lh": lh, "z": z, "u": u, "lx": log_x[:, -1], "f": exo[:, -1]}
        g_next = self.correction(exo[:, -1], z, lh)
        out["log_h_next"] = self._bound(self.garch_step(lh, log_x[:, -1], z, u) + g_next)
        out["g_next"] = g_next
        if squeeze:
            out = {k: v[0] for k, v in out.items()}
            new_state = {k: v[0] for k, v in new_state.items()}
        return out, new_state

    def _fused(self):
        """Recursion log h_{t+1} = a_t + b lh_t + kz z_t + ke (z_t^2 - 1) (+ g_t), with u_t
        substituted from the measurement equation (an exact rewrite of garch_step):
        a_t = A - Ku xi + (Cx + Ku) log x_t, b = Bh - Ku phi, kz = Kz - Ku d1, ke = Ke - Ku d2."""
        A, Bh, Cx, Kz, Ke, Ku = self.garch_coefs()
        return (A - Ku * self.xi, Cx + Ku, Bh - Ku * self.phi, Kz - Ku * self.d1, Ke - Ku * self.d2)

    def forward(self, r, log_x, exo=None, state=None):
        """Algorithm 1 on (B, L) or (L,) inputs.

        state: dict(lh, z, u, lx, f) at the time step preceding r[:, 0], or None to start.
        Returns (out, state) where out has (B, L) tensors log_h, z, u, g and
        out['log_h_next'] (B,) is the one-step forecast after the last observation.

        Speed (same maths as forward_reference, up to float rounding): everything that is
        not on the sequential path is vectorised (the log x / constant part of the recursion,
        the NN correction when it does not read the state, z and u after the loop); without
        autograd the loop runs in numpy (python floats for one stream).
        """
        squeeze, r, log_x, exo, lh, g0 = self._prep(r, log_x, exo, state)
        Bsz, L = r.shape
        a0, cx, b, kz, ke = self._fused()
        a_all = a0 + cx * log_x  # (B, L): a_t
        rc = r - self.mu
        state_nn = self.nn_active and self.nn_state_inputs
        need_z = self.step_uses_shock or state_nn
        g_pre = None  # g_t for all t, when it does not depend on the filtered state
        if not self.nn_active:
            g_pre = torch.zeros_like(r)
        elif not self.nn_state_inputs:  # g_{t+1} = s tanh(net(f_t)) for all t in one call
            g_lead = self.correction(exo, None, None)
            a_all = a_all + g_lead  # a_t enters log h_{t+1}
            g_pre = torch.cat([g0[:, None], g_lead[:, :-1]], 1)
        if state_nn:  # first layer: the exo part W1_f f_t + b1 for all t at once
            lin1 = self.net[0]
            nf = self.n_exo_nn
            pre1 = F.linear(exo[..., :nf], lin1.weight[:, :nf], lin1.bias)  # (B, L, H)
            w_z, w_h = lin1.weight[:, nf], lin1.weight[:, nf + 1]
        no_grad = not torch.is_grad_enabled() and r.device.type == "cpu"
        if no_grad:
            with np.errstate(over="ignore"):  # SiLU exp(-v) -> inf gives v/inf = -0, as torch
                lh_all, g_all = self._loop_numpy(lh, g0, a_all, rc, pre1 if state_nn else None,
                                                 b, kz, ke, need_z, state_nn)
            if g_pre is not None:
                g_all = g_pre
        else:
            lhs, gs = [lh], [g0]
            for t in range(1, L):
                x = a_all[:, t - 1] + b * lh
                if need_z:
                    z = rc[:, t - 1] * torch.exp(-0.5 * lh)
                    x = x + kz * z + ke * (z * z - 1)
                if state_nn:  # = self.correction(exo[:, t - 1], z, lh)
                    v = pre1[:, t - 1] + z[:, None] * w_z + lh[:, None] * w_h
                    g = self.nn_scale * torch.tanh(self.net[1:](v).squeeze(-1))
                    x = x + g
                    gs.append(g)
                lh = self._bound(x)
                lhs.append(lh)
            lh_all = torch.stack(lhs, 1)
            g_all = g_pre if g_pre is not None else torch.stack(gs, 1)
        return self._finish(squeeze, r, log_x, exo, lh_all, g_all)

    @torch.no_grad()
    def _loop_numpy(self, lh, g0, a_all, rc, pre1, b, kz, ke, need_z, state_nn):
        """The sequential loop of forward() without autograd (identical arithmetic)."""
        f = lambda v: float(v.detach())
        b, kz, ke, Bd, c = f(b), f(kz), f(ke), float(self.log_h_bound), f(self.lh_center)
        A = a_all.detach().cpu().numpy()
        R = rc.detach().cpu().numpy()
        Bsz, L = A.shape
        out = np.empty((Bsz, L))
        gout = np.empty((Bsz, L)) if state_nn else None
        if state_nn:
            lins = [m for m in self.net if isinstance(m, nn.Linear)]
            P1 = pre1.detach().cpu().numpy()
            nf = self.n_exo_nn
            wz, wh = (lins[0].weight[:, nf + i].detach().cpu().numpy() for i in (0, 1))
            W2, b2 = lins[1].weight.detach().cpu().numpy().T.copy(), lins[1].bias.detach().cpu().numpy()
            w3 = lins[2].weight.detach().cpu().numpy()[0]
            s = float(self.nn_scale)
            gout[:, 0] = g0.detach().cpu().numpy()
        lh0 = lh.detach().cpu().numpy()
        if Bsz == 1:  # python floats (and 1-d numpy for the NN): far less per-op overhead
            exp, log1p, tanh, dot = math.exp, math.log1p, math.tanh, np.dot
            sp = lambda v: (v if v > 0 else 0.0) + log1p(exp(-abs(v)))
            a, r_, h = A[0].tolist(), R[0].tolist(), float(lh0[0])
            res, gres = [h], []
            if state_nn:
                P, gres = P1[0], [float(gout[0, 0])]
            for t in range(1, L):
                x = a[t - 1] + b * h
                if need_z:
                    z = r_[t - 1] * exp(-0.5 * h)
                    x = x + kz * z + ke * (z * z - 1)
                if state_nn:
                    v = P[t - 1] + z * wz + h * wh
                    v = dot(v / (1 + np.exp(-v)), W2) + b2
                    g = s * tanh(float(dot(v / (1 + np.exp(-v)), w3)))
                    gres.append(g)
                    x = x + g
                d = x - c
                h = c + d - sp(d - Bd) + sp(-d - Bd)
                res.append(h)
            out[0] = res
            if state_nn:
                gout[0] = gres
        else:
            sp = lambda v: np.logaddexp(0.0, v)
            h = lh0.copy()
            out[:, 0] = h
            for t in range(1, L):
                x = A[:, t - 1] + b * h
                if need_z:
                    z = R[:, t - 1] * np.exp(-0.5 * h)
                    x = x + kz * z + ke * (z * z - 1)
                if state_nn:  # net = Linear, SiLU, Linear, SiLU, Linear(no bias)
                    v = P1[:, t - 1] + z[:, None] * wz + h[:, None] * wh
                    v = v / (1 + np.exp(-v)) @ W2 + b2
                    g = s * np.tanh((v / (1 + np.exp(-v))) @ w3)
                    gout[:, t] = g
                    x = x + g
                d = x - c
                h = c + d - sp(d - Bd) + sp(-d - Bd)
                out[:, t] = h
        tt = lambda v: torch.as_tensor(v, dtype=a_all.dtype, device=a_all.device)
        return tt(out), (tt(gout) if state_nn else None)

    def forward_reference(self, r, log_x, exo=None, state=None):
        """Original step-by-step Algorithm 1 (slow; kept as the reference for tests)."""
        squeeze, r, log_x, exo, lh, g0 = self._prep(r, log_x, exo, state)
        L = r.shape[1]
        mu, xi, phi = self.mu, self.xi, self.phi
        lhs, gs = [], []
        g = g0
        for t in range(L):
            if t > 0:
                g = self.correction(exo[:, t - 1], z, lh)
                lh = self._bound(self.garch_step(lh, log_x[:, t - 1], z, u) + g)
            z = (r[:, t] - mu) * torch.exp(-0.5 * lh)
            u = log_x[:, t] - xi - phi * lh - self.meas_lev(z)
            lhs.append(lh), gs.append(g)
        return self._finish(squeeze, r, log_x, exo, torch.stack(lhs, 1), torch.stack(gs, 1))

    # ------------------------------------------------------------ likelihood
    def nll_parts(self, out, mask=None):
        """Per-observation negative partial (return) and measurement log-likelihoods (eq. ll-joint)."""
        log_su2 = 2 * self.log_sigma_u
        l_r = 0.5 * (LOG2PI + out["log_h"] + out["z"] ** 2)
        if self.overnight and "on" in out:
            l_r = l_r + out["on"] * self.overnight_nll(out["log_h"], out["r_on"])
        l_x = 0.5 * (LOG2PI + log_su2 + out["u"] ** 2 / log_su2.exp())
        if mask is None:
            return l_r.mean(), l_x.mean()
        n = mask.sum()
        return (l_r * mask).sum() / n, (l_x * mask).sum() / n

    def overnight_nll(self, log_h, r_on):
        """-log N(r_on; 0, exp(kappa) h_t) (doc eq. on-lik); multiplied by the flag on_t."""
        lv = self.kappa + log_h
        return 0.5 * (LOG2PI + lv + r_on ** 2 * torch.exp(-lv))

    def _extra_params(self):
        return {"kappa": float(self.kappa.detach())} if self.overnight else {}

    # ------------------------------------------------------------ forecasting (Section 5, Alg. 3)
    @torch.no_grad()
    def set_empirical_mgf(self, z, u):
        """Store training residuals (z_hat, u_hat) for the non-parametric MGF (doc Section 5)."""
        ok = torch.isfinite(z) & torch.isfinite(u)
        self.emp_resid = (z[ok].detach().reshape(-1), u[ok].detach().reshape(-1))

    def _psi_state(self, c, A1, A2, Bu, su, mgf):
        if mgf == "empirical":
            z, u = self.emp_resid
            w = A1 * z + A2 * (z * z - 1) + Bu * u
            return torch.logsumexp(c[..., None] * w, -1) - math.log(w.numel())
        return psi(c, A1, A2, Bu, su)

    def _psi_meas(self, su, mgf, dtype):
        if mgf == "empirical":
            z, u = self.emp_resid
            return torch.logsumexp(self.meas_lev(z).detach() + u, 0) - math.log(z.numel())
        return psi(torch.tensor(1.0, dtype=dtype), self.d1.detach(), self.d2.detach(), 1.0, su)

    @torch.no_grad()
    def forecast(self, log_h1, horizon, g_bar=None, mgf="gaussian"):
        """Analytic k-step forecasts from log h_{t+1} (any shape S).

        m_1 = log h_{t+1} (already contains the known correction g_t);
        m_k = c + a m_{k-1} + g_bar for k >= 2, g_bar ~ E g (training mean; exact when g == 0)
        E h_{t+k}  = exp(m_k + sum_{j=0}^{k-2} psi(a^j))
        E x_{t+k}  = exp(xi + phi m_k + psi_meas(1) + sum_{j=0}^{k-2} psi(phi a^j))
        mgf: "gaussian" (closed form, z, u Gaussian) or "empirical" (training-residual means).
        For a PINN with g != 0 the closed form is an approximation for k >= 2.
        Returns tensors of shape (*S, horizon).
        """
        if mgf == "empirical" and self.emp_resid is None:
            raise ValueError("call set_empirical_mgf first")
        c, a, A1, A2, Bu = (v.detach() for v in self.state_space())
        su, xi, phi = self.sigma_u.detach(), self.xi.detach(), self.phi.detach()
        gb = torch.as_tensor(0.0 if g_bar is None else g_bar, dtype=log_h1.dtype)
        ms = [log_h1]
        for _ in range(horizon - 1):
            ms.append(c + a * ms[-1] + gb)
        m = torch.stack(ms, -1)
        self._check_guard(m)
        zero = m.new_zeros(1)
        if horizon >= 2:  # psi only at the exponents actually used: a^j, j = 0..k-2
            aj = a ** torch.arange(horizon - 1, dtype=m.dtype)
            cum_h = torch.cat([zero, torch.cumsum(self._psi_state(aj, A1, A2, Bu, su, mgf), 0)])
            cum_x = torch.cat([zero, torch.cumsum(self._psi_state(phi * aj, A1, A2, Bu, su, mgf), 0)])
        else:
            cum_h = cum_x = zero
        psi_meas = self._psi_meas(su, mgf, m.dtype)
        return {"log_h": m, "h": torch.exp(m + cum_h), "x": torch.exp(xi + phi * m + psi_meas + cum_x)}

    @torch.no_grad()
    def simulate_forecast(self, last_state, horizon, n_paths=5000, generator=None, exo_ctx=None):
        """Monte Carlo E[h_{t+k}], E[x_{t+k}] including the NN correction (Algorithm 4 style).

        exo_ctx (optional) evolves the NN inputs along each path: dict with
          lx_hist (W,) last observed log x (W >= max window), windows, mu, sd (feature
          standardisation from the training split), cal_ang (horizon,) calendar angle of
          t+1..t+horizon. Without it the inputs are frozen at their last value (approximate).
        """
        st = {k: v.expand(n_paths, *v.shape[-1:]) if k == "f" else v.expand(n_paths) for k, v in last_state.items()}
        hist = None
        if exo_ctx is not None and self.nn_active:
            hist = exo_ctx["lx_hist"].reshape(1, -1).expand(n_paths, -1).clone()
            W, mu, sd, ang = exo_ctx["windows"], exo_ctx["mu"], exo_ctx["sd"], exo_ctx["cal_ang"]

        def feats(j):
            if hist is None:
                return st["f"]
            har = [hist[:, -w:].mean(1) for w in W]
            a = ang[j].expand(n_paths)
            return (torch.stack(har + [torch.sin(a), torch.cos(a)], 1) - mu) / sd

        lh = self._bound(self.garch_step(st["lh"], st["lx"], st["z"], st["u"]) + self.correction(feats(0), st["z"], st["lh"]))
        hs, xs = [], []
        for j in range(horizon):
            z = torch.randn(n_paths, generator=generator, dtype=lh.dtype)
            u = torch.randn(n_paths, generator=generator, dtype=lh.dtype) * self.sigma_u
            lx = self.xi + self.phi * lh + self.meas_lev(z) + u
            hs.append(lh.exp()), xs.append(lx.exp())
            if hist is not None:
                hist = torch.cat([hist[:, 1:], lx[:, None]], 1)
            f = feats(min(j + 1, horizon - 1)) if hist is not None else st["f"]
            lh = self._bound(self.garch_step(lh, lx, z, u) + self.correction(f, z, lh))
        return {"h": torch.stack(hs, 1).mean(0), "x": torch.stack(xs, 1).mean(0)}

    @torch.no_grad()
    def simulate(self, T, seed=0, burn_in=500):
        """Simulate (r, log x, log h) from the pure model (NN off), Algorithm 4."""
        gen = torch.Generator().manual_seed(seed)
        c, a, *_ = self.state_space()
        lh = (c / (1 - a)).detach()
        r, lx, lhs = [], [], []
        for _ in range(T + burn_in):
            z = torch.randn((), generator=gen)
            u = torch.randn((), generator=gen) * self.sigma_u
            lxt = self.xi + self.phi * lh + self.meas_lev(z) + u
            r.append(self.mu + torch.exp(0.5 * lh) * z), lx.append(lxt), lhs.append(lh)
            lh = self._bound(self.garch_step(lh, lxt, z, u))
        s = lambda v: torch.stack(v)[burn_in:].detach()
        return s(r), s(lx), s(lhs)

    @torch.no_grad()
    def pinn_persistence(self, out, exo):
        """Reduced-form persistence of the fitted PINN: a + mean d g_{t-1} / d log h_{t-1}
        (to first order; the raw a is not the persistence once g depends on log h)."""
        if not (self.use_nn and self.nn_active and self.nn_state_inputs):
            return None
        with torch.enable_grad():
            lh = out["log_h"][:-1].detach().clone().requires_grad_(True)
            g = self.correction(exo[:-1], out["z"][:-1].detach(), lh)
            dg, = torch.autograd.grad(g.sum(), lh)
        return float(self.state_space()[1].detach() + dg.mean())

    def load_params(self, **p):
        """Set parameters from paper-notation values (used for simulation tests)."""
        raise NotImplementedError
