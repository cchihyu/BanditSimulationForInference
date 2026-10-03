"""Variance and sub-Gaussian proxy utilities (all quantities are variances)."""
import numpy as np
from scipy.special import logsumexp


def arm_values(value, k, name):
    out = np.broadcast_to(np.asarray(value, dtype=float), (k,)).copy()
    if np.any(~np.isfinite(out)) or np.any(out <= 0):
        raise ValueError(f"{name} must contain positive finite arm values")
    return out


def bernoulli_proxy(p):
    """Optimal centered Bernoulli proxy; continuous at p=1/2 and endpoints."""
    p = np.asarray(p, dtype=float)
    if np.any((p < 0) | (p > 1)):
        raise ValueError("Probabilities must be in [0, 1]")
    clipped = np.clip(p, 1e-15, 1-1e-15)
    delta = 1-2*clipped
    with np.errstate(divide='ignore', invalid='ignore'):
        value = delta / (2*(np.log1p(-clipped)-np.log(clipped)))
    value = np.where(np.abs(delta) < 1e-6, .25-delta**2/12, value)
    return np.where((p == 0) | (p == 1), 0., value)


def residual_proxy(residuals, alpha=.49, grid_size=4001, floor=1e-8):
    """Truncated empirical log-MGF estimate, not a certified upper bound."""
    z = np.asarray(residuals, dtype=float)
    if z.size < 2 or not 0 < alpha < .5 or grid_size < 3 or grid_size % 2 != 1:
        raise ValueError("Need >=2 residuals, 0<alpha<.5, and an odd grid >=3")
    z = z-z.mean()
    grid = np.linspace(-np.log(z.size)**alpha, np.log(z.size)**alpha, grid_size)
    best = max(float(np.mean(z*z)), floor)
    for chunk in np.array_split(grid[np.abs(grid) > 1e-5], 32):
        if chunk.size:
            values = 2*(logsumexp(chunk[:, None]*z, axis=1)-np.log(z.size))/chunk**2
            best = max(best, float(values.max()))
    return best
