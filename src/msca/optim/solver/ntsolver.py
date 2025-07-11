from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray


@dataclass
class NTResult:
    """Newton's solver result.

    Parameters
    ----------
    x
        The solution of the optimization.
    success
        Whether or not the optimizer exited successfully.
    fun
        The objective function value.
    grad
        Gradient of the objective function.
    hess
        Hessian of the objective function.
    niter
        Number of iterations.

    """

    x: NDArray
    success: bool
    fun: float
    grad: NDArray
    hess: NDArray
    niter: int


class NTSolver:
    """Newton's solver.

    Parameters
    ----------
    fun
        Optimization objective function
    grad
        Optimization gradient function
    hess
        Optimization hessian function

    """

    def __init__(self, fun: Callable, grad: Callable, hess: Callable):
        self.fun = fun
        self.grad = grad
        self.hess = hess

    def _update_params(
        self,
        x: list[NDArray],
        dx: list[NDArray],
        a_init: float = 1.0,
        a_const: float = 0.01,
        a_scale: float = 0.9,
        a_lb: float = 1e-3,
    ) -> tuple[float, list[NDArray]]:
        """Update parameters with line search.

        Parameters
        ----------
        x
            A list a parameters, including x, s, and v, where s is the slackness
            variable and v is the dual variable for the constraints.
        dx
            A list of direction for the parameters.
        a_init
            Initial step size, by default 1.0.
        a_const
            Constant for the line search condition, the larger the harder, by
            default 0.01.
        a_scale
            Shrinkage factor for step size, by default 0.9.
        a_lb
            Lower bound of the step size when the step size is below this bound
            the line search will be terminated.

        Returns
        -------
        float
            The step size in the given direction.

        """
        a = a_init
        x_next = x + a * dx
        g_next = self.grad(x_next)
        gnorm_curr = np.max(np.abs(self.grad(x)))
        gnorm_next = np.max(np.abs(g_next))

        while gnorm_next > (1 - a_const * a) * gnorm_curr:
            if a * a_scale < a_lb:
                break
            a *= a_scale
            x_next = x + a * dx
            g_next = self.grad(x_next)
            gnorm_next = np.max(np.abs(g_next))

        return a, x_next

    def minimize(
        self,
        x0: NDArray,
        xtol: float = 1e-8,
        gtol: float = 1e-8,
        max_iter: int = 100,
        a_init: float = 1.0,
        a_const: float = 0.01,
        a_scale: float = 0.9,
        a_lb: float = 1e-3,
        verbose: bool = False,
        mat_solve_method: str = "",
        mat_solve_options: dict | None = None,
    ) -> NDArray:
        """Minimize optimization objective over constraints.

        Parameters
        ----------
        x0
            Initial guess for the solution.
        xtol
            Tolerance for the differences in `x`, by default 1e-8.
        gtol
            Tolerance for the KKT system, by default 1e-8.
        max_iter
            Maximum number of iterations, by default 100.
        a_init
            Initial step size, by default 1.0.
        a_const
            Constant for the line search condition, the larger the harder, by
            default 0.01.
        a_scale
            Shrinkage factor for step size, by default 0.9.
        a_lb
            Lower bound of the step size when the step size is below this bound
            the line search will be terminated.
        verbose
            Indicator of if print out convergence history, by default False
        mat_solve_method
            Method to solve the linear system, by default "".
        mat_solve_options
            Options for the linear system solver, by default None.

        Returns
        -------
        NTResult
            Result of the solver.

        """

        # initialize the parameters
        x = x0.copy()
        mat_solve_options = mat_solve_options or {}

        g = self.grad(x)
        gnorm = np.max(np.abs(g))
        xdiff = 1.0
        step = 1.0
        niter = 0
        success = False

        if verbose:
            fun = self.fun(x)
            print(f"{type(self).__name__}:")
            print(
                f"{niter=:3d}, {fun=:.2e}, {gnorm=:.2e}, {xdiff=:.2e}, {step=:.2e}"
            )

        while (not success) and (niter < max_iter):
            niter += 1

            # compute all directions
            dx = -self.hess(x).solve(
                g, method=mat_solve_method, **mat_solve_options
            )

            # get step size
            step, x = self._update_params(x, dx, a_init, a_const, a_scale, a_lb)

            # update f and gnorm
            g = self.grad(x)
            gnorm = np.max(np.abs(g))
            xdiff = step * np.max(np.abs(dx))

            if verbose:
                fun = self.fun(x)
                print(
                    f"{niter=:3d}, {fun=:.2e}, {gnorm=:.2e}, {xdiff=:.2e}, {step=:.2e}"
                )
            success = gnorm <= gtol or xdiff <= xtol

        result = NTResult(
            x=x,
            success=success,
            fun=self.fun(x),
            grad=self.grad(x),
            hess=self.hess(x),
            niter=niter,
        )

        return result
