from typing import Callable

import numpy as np
from numpy.typing import NDArray


def armijo_line_search(
    gradient: Callable,
    x: NDArray,
    dx: NDArray,
    step_init: float = 1.0,
    step_const: float = 0.01,
    step_scale: float = 0.9,
    step_lb: float = 1e-3,
) -> float:
    """Armijo line search.

    Parameters
    ----------
    x
        A list a parameters, including x, s, and v, where s is the slackness
        variable and v is the dual variable for the constraints.
    dx
        A list of direction for the parameters.
    step_init
        Initial step size, by default 1.0.
    step_const
        Constant for the line search condition, the larger the harder, by
        default 0.01.
    step_scale
        Shrinkage factor for step size, by default 0.9.
    step_lb
        Lower bound of the step size when the step size is below this bound
        the line search will be terminated.

    Returns
    -------
    float
        The step size in the given direction.

    """
    step = step_init
    x_next = x + step * dx
    g_next = gradient(x_next)
    gnorm_curr = np.max(np.abs(gradient(x)))
    gnorm_next = np.max(np.abs(g_next))

    while gnorm_next > (1 - step_const * step) * gnorm_curr:
        if step * step_scale < step_lb:
            break
        step *= step_scale
        x_next = x + step * dx
        g_next = gradient(x_next)
        gnorm_next = np.max(np.abs(g_next))

    return step
