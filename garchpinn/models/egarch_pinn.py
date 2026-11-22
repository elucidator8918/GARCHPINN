"""EGARCHPINN: Realized EGARCH(1,1) of Hansen & Huang (2016) + neural correction.

log h_{t+1} = omega + beta log h_t + tau1 z_t + tau2 (z_t^2 - 1) + gamma u_t + s g(f_t)
log x_t     = xi + phi log h_t + delta1 z_t + delta2 (z_t^2 - 1) + u_t
beta is the persistence (parameterised through a scaled sigmoid).
"""
import math

import numpy as np
import torch
import torch.nn as nn

from .base import PERSIST_MAX, RealizedPINN, _logit


class EGARCHPINN(RealizedPINN):
    variant = "egarch"
    lev_names = ("t1", "t2")  # tau(z) in the GARCH equation

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        self.t1 = nn.Parameter(torch.tensor(-0.1))  # cpu-ok: module init
        self.t2 = nn.Parameter(torch.tensor(0.05))  # cpu-ok: module init
        self.gamma.data.fill_(0.3)
        self.omega.data.fill_(0.0)

    @property
    def beta(self):
        return self.persistence

    def garch_step(self, lh, lx, z, u):
        return self.omega + self.beta * lh + self.t1 * z + self.t2 * (z * z - 1) + self.gamma * u

    def garch_coefs(self):
        z0 = self.omega.new_zeros(())
        return self.omega, self.beta, z0, self.t1, self.t2, self.gamma

    def state_space(self):
        return self.omega, self.beta, self.t1, self.t2, self.gamma

    def e_log_abs_A(self, n=80):
        """E log|A(z)|, z ~ N(0,1), A(z) = beta - gamma phi - z/2 (tau1 - gamma delta1)
        - z^2 (tau2 - gamma delta2) = d log h_{t+1} / d log h_t given x (HH 2016 eq. 3).
        Filter is invertible/stable when this is < 0. Gauss-Hermite (probabilists')."""
        f = lambda v: float(v.detach())
        b, g, ph = f(self.beta), f(self.gamma), f(self.phi)
        z, w = np.polynomial.hermite_e.hermegauss(n)
        A = b - g * ph - 0.5 * z * (f(self.t1) - g * f(self.d1)) - z * z * (f(self.t2) - g * f(self.d2))
        return float((w * np.log(np.abs(A) + 1e-300)).sum() / w.sum())

    def params(self):
        f = lambda v: float(v.detach())
        b = f(self.beta)
        return dict(mu=f(self.mu), omega=f(self.omega), beta=b, gamma=f(self.gamma),
                    tau1=f(self.t1), tau2=f(self.t2), xi=f(self.xi), phi=f(self.phi),
                    delta1=f(self.d1), delta2=f(self.d2), sigma_u=f(self.sigma_u),
                    persistence=b, e_log_abs_A=self.e_log_abs_A(),
                    filter_stable=bool(self.e_log_abs_A() < 0),
                    half_life=math.log(0.5) / math.log(b) if 0 < b < 1 else float("inf"),
                    **self._extra_params())

    @torch.no_grad()
    def load_params(self, mu=0.0, omega=0.0, beta=0.97, gamma=0.3, tau1=-0.1, tau2=0.05,
                    xi=0.0, phi=1.0, delta1=-0.07, delta2=0.07, sigma_u=0.4, persistence=None, **_):
        b = beta if persistence is None else persistence
        for n, v in dict(mu=mu, omega=omega, gamma=gamma, t1=tau1, t2=tau2, xi=xi, phi=phi,
                         d1=delta1, d2=delta2).items():
            getattr(self, n).fill_(float(v))
        self.log_sigma_u.fill_(math.log(sigma_u))
        self.persist_raw.fill_(_logit(b / PERSIST_MAX))
        return self
