import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq


def capped_simplex(x: NDArray,
                   s: float,
                   lb: float | NDArray = 0.0,
                   ub: float | NDArray = 1.0) -> NDArray:
    x = np.asarray(x)
    if np.isscalar(lb):
        lb = np.repeat(lb, x.size)
    if np.isscalar(ub):
        ub = np.repeat(ub, x.size)

    if s < lb.sum() or s > ub.sum():
        raise ValueError("Cannot achieve the given sum by the given bounds.")

    def f(z):
        return np.sum(np.maximum(np.minimum(x - z, ub), lb)) - s

    a = x.min() - lb.min()
    b = x.max() + ub.max()

    z = brentq(f, a, b)
    return np.maximum(np.minimum(x - z, ub), lb)
