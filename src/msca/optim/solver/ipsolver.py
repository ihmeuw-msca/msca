from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from msca.array_interface import ArrayInterface
from msca.typing import Array, DenseArray


@dataclass
class IPResult:
    """Interior point solver result.

    Parameters
    ----------
    x
        The solution of the optimization.
    success
        Whether or not the optimizer exited successfully.
    objective
        The objective function value.
    gradient
        Gradient of the objective function.
    hessian
        Hessian of the objective function.
    niter
        Number of iterations.
    maxcv
        The maximum constraint violation.

    """

    x: DenseArray
    success: bool
    objective: float
    gradient: DenseArray
    hessian: Array
    niter: int
    maxcv: float


class IPSolver:
    """Interior point solver.

    Parameters
    ----------
    objective
        The optimization objective function.
    gradient
        The optimization gradient function.
    hessian
        The optimization hessian function.
    cmat
        The constraint linear mapping.
    cvec
        The constraint bounds.

    """

    def __init__(
        self,
        objective: Callable,
        gradient: Callable,
        hessian: Callable,
        cmat: Array,
        cvec: DenseArray,
        arrif: ArrayInterface,
    ):
        self.objective = objective
        self.gradient = gradient
        self.hessian = hessian
        self.cmat = cmat
        self.cvec = cvec
        self.arrif = arrif

    def get_kkt(self, p: list[DenseArray], m: float) -> list[DenseArray]:
        """Get the KKT system.

        Parameters
        ----------
        p
            A list a parameters, including x, s, and v, where s is the slackness
            variable and v is the dual variable for the constraints.
        m
            Interior point method barrier variable.

        Returns
        -------
        list[NDArray]
            The KKT system with three components.

        """
        return [
            self.cmat.dot(p[0]) + p[1] - self.cvec,
            p[1] * p[2] - m,
            self.gradient(p[0]) + self.cmat.T.dot(p[2]),
        ]

    def _update_params(
        self,
        p: list[DenseArray],
        dp: list[DenseArray],
        m: float,
        a_init: float = 1.0,
        a_const: float = 0.01,
        a_scale: float = 0.9,
        a_lb: float = 1e-3,
    ) -> tuple[float, list[DenseArray]]:
        """Update parameters with line search.

        Parameters
        ----------
        p
            A list a parameters, including x, s, and v, where s is the slackness
            variable and v is the dual variable for the constraints.
        dp
            A list of direction for the parameters.
        m
            Interior point method barrier variable.
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
        for i in [1, 2]:
            indices = dp[i] < 0.0
            if not any(indices):
                continue
            a = 0.99 * min(a, min(-p[i][indices] / dp[i][indices]))

        f_curr = self.get_kkt(p, m)
        p_next = [v.copy() for v in p]
        for i in range(len(p)):
            p_next[i] += a * dp[i]
        f_next = self.get_kkt(p_next, m)
        gnorm_curr = max(abs(self.arrif.hstack_dense_array(f_curr)))
        gnorm_next = max(abs(self.arrif.hstack_dense_array(f_next)))

        while gnorm_next > (1 - a_const * a) * gnorm_curr:
            if a * a_scale < a_lb:
                break
            a *= a_scale
            p_next = [v.copy() for v in p]
            for i in range(len(p)):
                p_next[i] += a * dp[i]
            f_next = self.get_kkt(p_next, m)
            gnorm_next = max(abs(self.arrif.hstack_dense_array(f_next)))

        return a, p_next

    def minimize(
        self,
        x0: DenseArray,
        xtol: float = 1e-8,
        gtol: float = 1e-8,
        mtol: float = 1e-6,
        max_iter: int = 100,
        m_init: float = 1.0,
        m_freq: int = 5,
        m_scale: float = 0.5,
        a_init: float = 1.0,
        a_const: float = 0.01,
        a_scale: float = 0.9,
        a_lb: float = 1e-3,
        verbose: bool = False,
        mat_solve_method: str = "",
        mat_solve_options: dict | None = None,
    ) -> IPResult:
        """Minimize optimization objective over constraints.

        Parameters
        ----------
        x0
            Initial guess for the solution.
        xtol
            Tolerance for the differences in `x`, by default 1e-8.
        gtol
            Tolerance for the KKT system, by default 1e-8.
        mtol
            Tolerance for the log barrier parameter m, by default 1e-6.
        max_iter
            Maximm number of iterations, by default 100.
        m_init
            Initial interior point bairrier parameter, by default 1.0.
        m_freq
            Parameter m updating frequency, by default 5.
        m_scale
            Shrinkage factor for m updates, by default 0.1
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
        IPResult
            The result of the solver.

        """

        # initialize the parameters
        p = [
            x0,
            self.arrif.as_dense_array(np.ones(self.cvec.size)),
            self.arrif.as_dense_array(np.ones(self.cvec.size)),
        ]
        mat_solve_options = mat_solve_options or {}

        m = m_init
        f = self.get_kkt(p, m)
        gnorm = max(abs(self.arrif.hstack_dense_array(f)))
        xdiff = 1.0
        step = 1.0
        niter = 0
        success = False

        if verbose:
            objective = self.objective(p[0])
            print(f"{type(self).__name__}:")
            print(
                f"{niter=:3d}, {objective=:.2e}, {gnorm=:.2e}, {xdiff=:.2e}, {step=:.2e}, {m=:.2e}"
            )

        while (not success) and (niter < max_iter):
            niter += 1

            # cache convenient variables
            sv_vec = p[2] / p[1]
            sf2_vec = f[1] / p[1]
            csv_mat = self.arrif.scale_rows(self.cmat, sv_vec)

            # compute all directions
            mat = self.hessian(p[0]) + csv_mat.T.dot(self.cmat)
            vec = -f[2] + self.cmat.T.dot(sf2_vec - sv_vec * f[0])
            dx = self.arrif.solve(
                mat, vec, method=mat_solve_method, **mat_solve_options
            )
            ds = -f[0] - self.cmat.dot(dx)
            dv = -sf2_vec - sv_vec * ds
            dp = [dx, ds, dv]

            # get step size
            step, p = self._update_params(p, dp, m, a_init, a_const, a_scale, a_lb)

            # update m
            if niter % m_freq == 0:
                m = max(m_scale * m, 0.1 * p[1].dot(p[2]) / len(p[1]))

            # update f and gnorm
            f = self.get_kkt(p, m)
            gnorm = max(abs(self.arrif.hstack_dense_array(f)))
            xdiff = step * max(abs(dp[0]))

            if verbose:
                objective = self.objective(p[0])
                print(
                    f"{niter=:3d}, {objective=:.2e}, {gnorm=:.2e}, {xdiff=:.2e}, {step=:.2e}, {m=:.2e}"
                )
            success = (gnorm <= gtol or xdiff <= xtol) and (m <= mtol)

        maxcv = max(max(self.cmat.dot(p[0]) - self.cvec), 0.0)
        result = IPResult(
            x=p[0],
            success=success,
            objective=self.objective(p[0]),
            gradient=self.gradient(p[0]),
            hessian=self.hessian(p[0]),
            niter=niter,
            maxcv=float(maxcv),
        )

        return result
