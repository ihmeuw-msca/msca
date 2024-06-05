from collections import deque
from dataclasses import dataclass
from functools import partial
from typing import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator, cg

from msca.linalg.matrix import Matrix
from msca.optim.line_search import line_search_map
from msca.optim.precon import precon_builder_map


@dataclass
class NTCGResult:
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


class NTCGSolver:
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
        self.record = {}

    def _build_precon(
        self, s_deque: deque, y_deque: deque, r_deque: deque, hess: Matrix
    ) -> LinearOperator | None:
        if len(s_deque) == 0:
            return None

        gamma = 1 / (r_deque[-1] * np.dot(y_deque[-1], y_deque[-1]))

        def precon_mv(x):
            q = x.copy()
            a_deque = deque()
            for s, y, r in list(zip(s_deque, y_deque, r_deque))[::-1]:
                a = r * s.dot(q)
                q -= a * y
                a_deque.append(a)
            z = gamma * q
            for s, y, r in zip(s_deque, y_deque, r_deque):
                b = r * y.dot(z)
                z += (a_deque.pop() - b) * s
            return z

        precon = LinearOperator(hess.shape, matvec=precon_mv)

        return precon

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
        verbose
            Indicator of if print out convergence history, by default False
        cg_options
            Options for the linear system solver, by default None.

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
        cg_options = cg_options or {}

        if cg_maxiter_init is None and cg_maxiter is None:

            def get_cg_maxiter(niter):
                return None

        elif cg_maxiter is None:

            def get_cg_maxiter(niter):
                return cg_maxiter_init + cg_maxiter_incr * (niter - 1)

        else:

            def get_cg_maxiter(niter):
                return min(cg_maxiter, cg_maxiter_init + cg_maxiter_incr * (niter - 1))

        g = self.grad(x)
        gnorm = np.max(np.abs(g))
        xdiff = 1.0
        step = 1.0
        niter = 0
        success = False

        x_pair = deque([x], maxlen=2)
        g_pair = deque([g], maxlen=2)

        if verbose:
            fun = self.fun(x)
            print(f"{type(self).__name__}:")
            print(f"{niter=:3d}, {fun=:.2e}, {gnorm=:.2e}, {xdiff=:.2e}, {step=:.2e}")

        while (not success) and (niter < maxiter):
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
            # gnorm = np.max(np.abs(g))
            dx = cg(hess, -g, **cg_options)[0]
            # dx *= gnorm

            # get step size
            step = line_search(self.grad, x, dx, **line_search_options)
            x = x + step * dx

            # update f and gnorm
            g = self.grad(x)
            gnorm = np.max(np.abs(g))
            xdiff = step * np.max(np.abs(dx))

            x_pair.append(x)
            g_pair.append(g)

            if verbose:
                fun = self.fun(x)
                print(
                    f"{niter=:3d}, {fun=:.2e}, {gnorm=:.2e}, {xdiff=:.2e}, "
                    f"{step=:.2e}, cg_iter={cg_info['iter']}"
                )
            success = gnorm <= gtol or xdiff <= xtol

        result = NTCGResult(
            x=x,
            success=success,
            fun=self.fun(x),
            grad=self.grad(x),
            hess=self.hess(x),
            niter=niter,
        )

        return result
