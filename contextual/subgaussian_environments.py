"""True contextual environments; noise shapes are fixed within each arm."""
import numpy as np
from scipy.special import expit
from .dispersion import arm_values


class SubGaussianEnvironment:
    def __init__(self, kind, beta, scales=1., half_widths=1., mixtures=None):
        if kind not in {'scaled_bernoulli', 'gaussian_mixture', 'uniform'}:
            raise ValueError('Unsupported reward environment')
        self.kind = kind
        self.beta = np.asarray(beta, dtype=float)
        if self.beta.ndim != 2 or self.beta.shape[1] < 1 or not np.isfinite(self.beta).all():
            raise ValueError('beta must be a finite (K, d+1) matrix')
        self.n_actions, p = self.beta.shape
        self.context_dim = p-1
        self.scales = arm_values(scales, self.n_actions, 'scales')
        self.half_widths = arm_values(half_widths, self.n_actions, 'half_widths')
        if mixtures is None:
            mixtures = [dict(weights=[.7,.3], means=[-.5, 7/6], sigmas=[.4,1.]) for _ in range(self.n_actions)]
        if len(mixtures) != self.n_actions:
            raise ValueError('Supply one mixture specification per arm')
        self.mixtures = []
        for mix in mixtures:
            w, m, s = (np.asarray(mix[key], dtype=float) for key in ('weights','means','sigmas'))
            if (w.ndim != 1 or w.size == 0 or w.shape != m.shape or w.shape != s.shape
                    or not np.isfinite(np.r_[w,m,s]).all() or np.any(w <= 0) or np.any(s <= 0)):
                raise ValueError('Invalid Gaussian mixture parameters')
            w = w/w.sum(); m = m-w@m
            self.mixtures.append((w,m,s))

    def mean(self, x, a, params=None):
        eta = np.r_[1., x] @ self.beta[a]
        return float(self.scales[a]*expit(eta) if self.kind == 'scaled_bernoulli' else eta)

    def sample(self, x, a, rng, params=None):
        mu = self.mean(x,a)
        if self.kind == 'scaled_bernoulli':
            return float(self.scales[a]*rng.binomial(1,mu/self.scales[a]))
        if self.kind == 'uniform':
            return float(mu+rng.uniform(-self.half_widths[a],self.half_widths[a]))
        w,m,s = self.mixtures[a]; j = rng.choice(len(w),p=w)
        return float(mu+rng.normal(m[j],s[j]))

    def support_widths(self):
        if self.kind == 'gaussian_mixture':
            raise ValueError('Hoeffding is unavailable for unbounded Gaussian mixtures')
        return self.scales if self.kind == 'scaled_bernoulli' else 2*self.half_widths
