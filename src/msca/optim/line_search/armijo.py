from typing import Callable

import numpy as np
from numpy.typing import NDArray

def armijo_line_search(
    x,
    p,
    g,
    objective: Callable,
    step_init: float = 1.0,
    alpha: float = 0.01,
    shrinkage: float = 0.5,
):
    """
    Performs an Armijo line search to select an appropriate step size along a given search direction.
    This function iteratively reduces the step size until the decrease in the objective function, along the direction of descent,
    satisfies the Armijo (sufficient decrease) condition. In each iteration, it checks whether the new point yields a value that is
    lower than the current value by a margin proportional to the step and directional derivative. If no satisfactory step size is found
    and the step size becomes exceedingly small (<= 1e-15), a RuntimeError is raised.
    Parameters:
        x (array_like): The current point or position in the parameter space.
        p (array_like): The descent direction along which the line search is performed.
        g (array_like): The gradient of the objective function evaluated at x.
        objective (Callable): A callable that computes the objective function value given a point.
        step_init (float, optional): The initial step size to start the line search. Default is 1.0.
        alpha (float, optional): The Armijo condition control parameter defining the sufficient decrease criterion. Default is 0.01.
        shrinkage (float, optional): The factor by which the step is multiplied to reduce the step size in each iteration. Default is 0.5.
    Returns:
        float: The step size that satisfies the Armijo sufficient decrease condition.
    Raises:
        RuntimeError: If the step size becomes too small (<= 1e-15) without satisfying the Armijo condition,
                      indicating failure in finding a suitable step size.
    """
    def sufficiently_improved(new_val, step):
        return (new_val - val <= -1 * alpha * step * np.dot(g, p)) and (
            not np.isnan(new_val)
        )

    step = step_init
    new_x = x - step * p
    val, new_val = objective(x), objective(new_x)
    while (not sufficiently_improved(new_val, step)):
        if step <= 1e-15:
            raise RuntimeError(
                f"Line Search Failed, new_val = {new_val}, prev_val = {val}"
            )
        step *= shrinkage
        new_x = x - step * p
        new_val = objective(new_x)
    return step