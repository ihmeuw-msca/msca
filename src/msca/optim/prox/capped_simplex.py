from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq


def proj_capped_simplex(
    x: NDArray, s: float, lb: float | NDArray = 0.0, ub: float | NDArray = 1.0
) -> NDArray:
    """Projection onto the capped simplex. The capped simplex can be defined as

    .. math::
        \\triangle(s, l, u) =
        \\left\\{x \\in \\mathbb{R}^n :
        l \\le x \\le u, \\sum_{i=1}^n x_i = s\\right\\}

    And the projection problem can be frame as an optimization problem.

    .. math::

        \\min_{y} \\frac{1}{2}\\|y - x\\|^2, \\quad
        \\mathrm{s.t.}\\; y\\in\\triangle(s, l, u)

    The solution can be found by solving the following equation for :math:`z`.

    .. math::

        \\mathbf{1}^\\top \\max(\\min(x - z \\mathbf{1}, u), l) = s

    Given :math:`z` the solution of the projection problem is
    :math:`y = \\max(\\min(x - z \\mathbf{1}, u), l)`.

    Parameters
    ----------
    x
        The vector that needs to be projected.
    s
        The target sum of the vector.
    lb
        The lower bounds of the target variable. If it is a scalar, it will be
        used for all coordinates of the variable. Default is 0.
    ub
        The lower bounds of the target variable. If it is a scalar, it will be
        used for all coordinates of the variable. Default is 1.

    Returns
    -------
    NDArray
        The target variable that satisfy the bounds and sum constraints.

    Raises
    ------
    ValueError
        Raised when the provided sum is lower than sum of the lower bounds or
        higher than the sum of the upper bounds.

    """
    x = np.asarray(x)
    if np.isscalar(lb):
        lb = np.repeat(lb, x.size)
    if np.isscalar(ub):
        ub = np.repeat(ub, x.size)

    if s < lb.sum() or s > ub.sum():
        raise ValueError("Cannot achieve the given sum by the given bounds.")

    def f(z):
        return np.sum(np.maximum(np.minimum(x - z, ub), lb)) - s

    a = (x - ub).min()
    b = (x - lb).max()

    z = brentq(f, a, b)
    return np.maximum(np.minimum(x - z, ub), lb)
