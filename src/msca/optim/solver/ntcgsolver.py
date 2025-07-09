from collections import deque
from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import cg

from msca.optim.line_search import line_search_map
from msca.optim.precon import precon_builder_map


@dataclass
class NTCGResult:
    """Newton CG's solver result.

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


class NTCGSolver:
    """Newton CG's solver.

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

    def minimize(
        self,
        x0: NDArray,
        xtol: float = 1e-8,
        gtol: float = 1e-8,
        maxiter: int = 100,
        line_search: str = "armijo",
        line_search_options: dict | None = None,
        precon_builder: str | None = None,
        precon_builder_options: dict | None = None,
        cg_maxiter_init: int | None = None,
        cg_maxiter_incr: int = 0,
        cg_maxiter: int | None = None,
        cg_options: dict | None = None,
        verbose: bool = False,
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
        maxiter
            Maximum number of iterations, by default 100.
        line_search
            Line search method, by default "armijo".
        line_search_options
            Options for the line search method, by default None.
        precon_builder
            Preconditioner builder, by default None.
        precon_builder_options
            Options for the preconditioner builder, by default None.
        cg_maxiter_init
            Initial maximum number of CG iterations, by default None. If it is
            None, solver will try to use cg_maxiter as a constant cap of number
            of CG iterations. And if cg_maxiter is also None, there will be no
            cap.
        cg_maxiter_incr
            Increment of maximum number of CG iterations, by default 0. After
            the increment, the number of CG iterations will still be capped by
            cg_maxiter if cg_maxiter is not None.
        cg_maxiter
            Maximum number of CG iterations, by default None.
        cg_options
            Options for the CG function, by default None.
        verbose
            Indicator of if print out convergence history, by default False

        Returns
        -------
        NTCGResult
            Result of the solver.

        """

        # initialize the parameters
        x = x0.copy()
        line_search = line_search_map[line_search]
        line_search_options = line_search_options or {}
        if precon_builder is not None:
            precon_builder = precon_builder_map[precon_builder](
                **(precon_builder_options or {})
            )
        cg_options = cg_options or {"rtol": 1e-2}

        def get_cg_maxiter(niter: int) -> int | None:
            if cg_maxiter_init is None and cg_maxiter is None:
                return None
            if cg_maxiter_init is None:
                return cg_maxiter
            result = cg_maxiter_init + cg_maxiter_incr * (niter - 1)
            if cg_maxiter is not None:
                result = min(result, cg_maxiter)
            return result

        g = self.grad(x)
        gnorm = np.max(np.abs(g))
        xdiff = 1.0
        step = 1.0
        niter = 0
        success = False
        failure = False

        x_pair = deque([x], maxlen=2)
        g_pair = deque([g], maxlen=2)

        if verbose:
            fun = self.fun(x)
            print(f"{type(self).__name__}:")
            print(
                f"{niter=:3d}, {fun=:.2e}, {gnorm=:.2e}, {xdiff=:.2e}, {step=:.2e}"
            )

        while (not success) and (not failure) and (niter < maxiter):
            niter += 1

            # compute all directions
            cg_info = dict(iter=0)

            def cg_iter_counter(xk, cg_info):
                cg_info["iter"] += 1

            hess = self.hess(x)

            cg_options["callback"] = partial(cg_iter_counter, cg_info=cg_info)
            if precon_builder is not None:
                cg_options["M"] = precon_builder(x_pair, g_pair)
            cg_options["maxiter"] = get_cg_maxiter(niter)
            dx = cg(hess, -g, **cg_options)[0]
            try:
                # get step size
                step = line_search(
                    gradient=self.grad, x=x, dx=-dx, **line_search_options
                )
            except RuntimeError:
                dx = -g
                step = line_search(
                    gradient=self.grad, x=x, dx=-dx, **line_search_options
                )
            x = x + step * dx

            # update f and gnorm
            g = self.grad(x)
            gnorm = np.max(np.abs(g))
            xdiff = step * np.max(np.abs(dx))

            x_pair.append(x)
            g_pair.append(g)

            fun = self.fun(x)
            if verbose:
                print(
                    f"{niter=:3d}, {fun=:.2e}, {gnorm=:.2e}, {xdiff=:.2e}, "
                    f"{step=:.2e}, cg_iter={cg_info['iter']}"
                )
            success = gnorm <= gtol or xdiff <= xtol
            failure = not (
                np.isfinite(fun)
                and np.isfinite(gnorm)
                and np.isfinite(xdiff)
                and np.isfinite(step)
            )

        result = NTCGResult(
            x=x,
            success=success,
            fun=self.fun(x),
            grad=self.grad(x),
            hess=self.hess(x),
            niter=niter,
        )

        return result
