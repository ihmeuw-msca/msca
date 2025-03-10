from collections import deque

import numpy as np
from numpy.typing import NDArray
from scipy.sparse.linalg import LinearOperator


class LBFGSPreconBuilder:
    """LBFGS preconditioner builder. This is based on the LBFGS algorithm
    `here <https://en.wikipedia.org/wiki/Limited-memory_BFGS>`_. We use the
    built linear operator as the preconditioner of the system.

    Parameters
    ----------
    rank
        Rank of the LBFGS approximation. Default is 5.

    """

    def __init__(self, rank: int = 5) -> None:
        self.rank = rank
        self._s = deque(maxlen=self.rank)
        self._y = deque(maxlen=self.rank)
        self._r = deque(maxlen=self.rank)

    def __call__(
        self, x_pair: deque[NDArray], g_pair: deque[NDArray]
    ) -> LinearOperator | None:
        """Returns the preconditioner.

        Parameters
        ----------
        x_pair
            The previous and current variables.
        g_pair
            The gradient at previous and current variables.

        Returns
        -------
        LinearOperator | None
            If `x_pair` or `g_pair` only contains information of one iteration
            it will return None which means no preconditioning. Because we
            cannot compute the LBFGS inverse Hessian approximation based on one
            iteration. Otherwise it will return a linear operator as the LBFGS
            inverse Hessian approximation.

        """
        if len(x_pair) != 2 or len(g_pair) != 2:
            return
        self._s.append(x_pair[1] - x_pair[0])
        self._y.append(g_pair[1] - g_pair[0])
        self._r.append(1 / np.dot(self._s[-1], self._y[-1]))

        size = len(self._s[0])
        gamma = 1 / (self._r[-1] * np.dot(self._y[-1], self._y[-1]))
        iterator = list(zip(self._s, self._y, self._r))

        def precon_mv(x):
            q = x.copy()
            a_deque = deque()
            for s, y, r in reversed(iterator):
                a = r * s.dot(q)
                q -= a * y
                a_deque.append(a)
            z = gamma * q
            for s, y, r in iterator:
                b = r * y.dot(z)
                z += (a_deque.pop() - b) * s
            return z

        precon = LinearOperator(
            (size, size), matvec=precon_mv, dtype=x_pair[0].dtype
        )

        return precon
