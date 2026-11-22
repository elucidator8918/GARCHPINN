"""GARCHPINN: log-linear Realized GARCH(1,1) of Hansen, Huang & Shek (2012) + neural correction.

log h_{t+1} = omega + beta log h_t + gamma log x_t + s g(f_t)
log x_t     = xi + phi log h_t + d1 z_t + d2 (z_t^2 - 1) + u_t
Persistence pi = beta + gamma phi is the parameterised quantity, beta = pi - gamma phi.
"""
import math

import torch

from .base import PERSIST_MAX, RealizedPINN, _logit


class GARCHPINN(RealizedPINN):
    variant = "garch"

    @property
    def beta(self):
        return self.persistence - self.gamma * self.phi

    def garch_step(self, lh, lx, z, u):
        return self.omega + self.beta * lh + self.gamma * lx

    step_uses_shock = False

    def garch_coefs(self):
        z0 = self.omega.new_zeros(())
        return self.omega, self.beta, self.gamma, z0, z0, z0

    def state_space(self):
        g = self.gamma
        return self.omega + g * self.xi, self.persistence, g * self.d1, g * self.d2, g

    def params(self):
        f = lambda v: float(v.detach())
        pi = f(self.persistence)
        return dict(mu=f(self.mu), omega=f(self.omega), beta=f(self.beta), gamma=f(self.gamma),
                    xi=f(self.xi), phi=f(self.phi), tau1=f(self.d1), tau2=f(self.d2),
                    sigma_u=f(self.sigma_u), persistence=pi,
                    # filter stability (HHS Lemma 1): |beta| < 1 besides |pi| < 1
                    filter_stable=bool(abs(f(self.beta)) < 1),
                    half_life=math.log(0.5) / math.log(pi) if 0 < pi < 1 else float("inf"),
                    **self._extra_params())

    @torch.no_grad()
    def load_params(self, mu=0.0, omega=0.0, beta=None, gamma=0.4, xi=0.0, phi=1.0,
                    tau1=-0.07, tau2=0.07, sigma_u=0.4, persistence=None, **_):
        if persistence is None:
            if beta is None:
                raise ValueError("give beta or persistence")
            persistence = beta + gamma * phi
        for n, v in dict(mu=mu, omega=omega, gamma=gamma, xi=xi, phi=phi, d1=tau1, d2=tau2).items():
            getattr(self, n).fill_(float(v))
        self.log_sigma_u.fill_(math.log(sigma_u))
        self.persist_raw.fill_(_logit(persistence / PERSIST_MAX))
        return self
