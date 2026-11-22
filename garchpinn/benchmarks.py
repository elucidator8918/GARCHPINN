"""Benchmarks for the evaluation protocol (reference doc Section 8).

GARCH(1,1) on returns (eq. garch), fitted by Gaussian QMLE with torch L-BFGS:
    h_{t+1} = omega + alpha eps_t^2 + beta h_t,  eps_t = r_t - mu
    omega = softplus(.), alpha + beta = P sigmoid(.) < 1, alpha = (alpha+beta) sigmoid(.)
    E_t h_{t+k} = s2 + (alpha+beta)^{k-1} (h_{t+1} - s2),  s2 = omega / (1 - alpha - beta)
Log-HAR (Corsi 2009, eq. har) direct-horizon regressions:
    log( (1/k) sum_{i=1..k} x_{t+i} ) = b' [1, HAR averages of log x up to t] + e
    level forecast  exp(fit + s^2/2)   (log-normal bias correction)
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

PMAX = 0.9999


def _inv_softplus(y):
    return math.log(math.expm1(y))


def _logit(p):
    return math.log(p / (1 - p))


class GARCH11:
    def __init__(self):
        self.mu = 0.0
        self.omega = self.alpha = self.beta = None
        self.h0 = None

    # ---- filter
    @staticmethod
    def _filter(r, mu, omega, alpha, beta, h0):
        """h[t] = variance of r_t given F_{t-1}; also returns h_next after the last obs."""
        eps2 = (r - mu) ** 2
        h = [h0]
        for t in range(len(r)):
            h.append(omega + alpha * eps2[t] + beta * h[-1])
        h = torch.stack(h)
        return h[:-1], h[-1]

    def _unpack(self, th):
        omega = F.softplus(th[1])
        pers = PMAX * torch.sigmoid(th[2])
        alpha = pers * torch.sigmoid(th[3])
        return th[0], omega, alpha, pers - alpha

    def fit(self, r, iters=200):
        # always fitted on CPU: the sequential python loop + autograd is launch-bound on a GPU
        r = torch.as_tensor(r).detach().cpu()  # cpu-ok: deliberate CPU fit
        dt = r.dtype
        self.h0 = r[: min(22, len(r))].var(unbiased=False).clamp_min(1e-12)
        v = r.var(unbiased=False).item()
        th = torch.tensor([r.mean().item(), _inv_softplus(max(v * 0.03, 1e-8)), _logit(0.97 / PMAX),
                           _logit(0.08 / 0.97)], dtype=dt, device=r.device, requires_grad=True)
        opt = torch.optim.LBFGS([th], lr=1.0, max_iter=iters, line_search_fn="strong_wolfe",
                                tolerance_grad=1e-9, tolerance_change=1e-12)

        def closure():
            opt.zero_grad()
            mu, om, a, b = self._unpack(th)
            h, _ = self._filter(r, mu, om, a, b, self.h0)
            nll = 0.5 * (math.log(2 * math.pi) + torch.log(h) + (r - mu) ** 2 / h).mean()
            nll.backward()
            return nll

        opt.step(closure)
        with torch.no_grad():
            mu, om, a, b = self._unpack(th)
        self.mu, self.omega, self.alpha, self.beta = mu.item(), om.item(), a.item(), b.item()
        return self

    @torch.no_grad()
    def filter(self, r):
        """Returns h of length T+1: h[t] = E_{t-1} r_t^2 (centered), h[T] = forecast after last obs."""
        r = torch.as_tensor(r)  # cpu-ok: keeps the input's device
        dev = r.device
        h, hn = self._filter(r.cpu(), self.mu, self.omega, self.alpha, self.beta, self.h0.cpu())
        return torch.cat([h, hn[None]]).to(dev)

    @property
    def uncond_var(self):
        return self.omega / (1 - self.alpha - self.beta)

    def forecast(self, h1, horizon):
        """E h_{t+k}, k=1..horizon, from h_{t+1} (any shape) -> (*S, horizon)."""
        h1 = torch.as_tensor(h1)  # cpu-ok: keeps the input's device
        p = (self.alpha + self.beta) ** torch.arange(horizon, dtype=h1.dtype, device=h1.device)
        s2 = self.uncond_var
        return s2 + p * (h1[..., None] - s2)

    def params(self):
        return {"mu": self.mu, "omega": self.omega, "alpha": self.alpha, "beta": self.beta,
                "persistence": self.alpha + self.beta}


class LogHAR:
    """Direct-horizon log-HAR. Origin t uses HAR averages of log x_{t-w+1..t}."""

    def __init__(self, windows=(1, 5, 22)):
        self.windows = tuple(windows)
        self.coef, self.s2 = {}, {}

    def features(self, log_x):
        lx = np.asarray(log_x, dtype=float)
        cs = np.concatenate([[0.0], np.cumsum(lx)])
        T = len(lx)
        cols = [np.ones(T)]
        for w in self.windows:
            idx = np.arange(T)
            lo = np.maximum(idx + 1 - w, 0)
            cols.append((cs[idx + 1] - cs[lo]) / (idx + 1 - lo))
        return np.column_stack(cols)

    @staticmethod
    def target(log_x, k):
        """y[t] = log mean x_{t+1..t+k}; NaN where the window runs off the end."""
        x = np.exp(np.asarray(log_x, dtype=float))
        cs = np.concatenate([[0.0], np.cumsum(x)])
        T = len(x)
        y = np.full(T, np.nan)
        t = np.arange(T - k)
        y[t] = np.log((cs[t + k + 1] - cs[t + 1]) / k)
        return y

    def fit(self, log_x, n_fit, horizons):
        """OLS on origins t with t+k < n_fit (targets stay inside the fit split)."""
        lx = np.asarray(log_x, dtype=float)
        X = self.features(lx)
        burn = max(self.windows) - 1
        for k in horizons:
            y = self.target(lx, k)
            t = np.arange(burn, n_fit - k)
            b, *_ = np.linalg.lstsq(X[t], y[t], rcond=None)
            res = y[t] - X[t] @ b
            self.coef[k] = b
            self.s2[k] = float(res @ res / max(len(t) - X.shape[1], 1))
        return self

    def predict_sum(self, log_x, k):
        """Forecast of sum_{i=1..k} x_{t+i} for every origin t (length T)."""
        X = self.features(log_x)
        return k * np.exp(X @ self.coef[k] + 0.5 * self.s2[k])
