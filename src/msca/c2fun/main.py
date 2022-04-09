from __future__ import annotations

from abc import ABC, abstractproperty, abstractstaticmethod
from typing import Dict

import numpy as np
from numpy.typing import NDArray


class C2Fun(ABC):
    """Continuously twice differentiable function.

    """

    @abstractproperty
    def inv(self) -> C2Fun:
        pass

    @abstractstaticmethod
    def _fun(x: NDArray) -> NDArray:
        pass

    @abstractstaticmethod
    def _dfun(x: NDArray) -> NDArray:
        pass

    @abstractstaticmethod
    def _d2fun(x: NDArray) -> NDArray:
        pass

    def __call__(self, x: NDArray, order: int = 0) -> NDArray:
        if order == 0:
            return self._fun(x)
        if order == 1:
            return self._dfun(x)
        if order == 2:
            return self._d2fun(x)
        raise ValueError("Order has to be selected from 0, 1 or 2.")

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Identity(C2Fun):
    """Identity function.

    """

    @property
    def inv(self) -> C2Fun:
        return self

    @staticmethod
    def _fun(x: NDArray) -> NDArray:
        x = np.asarray(x)
        return x

    @staticmethod
    def _dfun(x: NDArray) -> NDArray:
        x = np.asarray(x)
        return np.ones(x.size, dtype=x.dtype)

    @staticmethod
    def _d2fun(x: NDArray) -> NDArray:
        x = np.asarray(x)
        return np.zeros(x.size, dtype=x.dtype)


class Exp(C2Fun):
    """Exponential function.

    """

    @property
    def inv(self) -> C2Fun:
        return log

    @staticmethod
    def _fun(x: NDArray) -> NDArray:
        return np.exp(x)

    @staticmethod
    def _dfun(x: NDArray) -> NDArray:
        return np.exp(x)

    @staticmethod
    def _d2fun(x: NDArray) -> NDArray:
        return np.exp(x)


class Log(C2Fun):
    """Log function.

    """

    @property
    def inv(self) -> C2Fun:
        return exp

    @staticmethod
    def _fun(x: NDArray) -> NDArray:
        return np.log(x)

    @staticmethod
    def _dfun(x: NDArray) -> NDArray:
        x = np.asarray(x)
        return 1 / x

    @staticmethod
    def _d2fun(x: NDArray) -> NDArray:
        x = np.asarray(x)
        return -1 / x**2


class Expit(C2Fun):
    """Expit function.

    """

    @property
    def inv(self) -> C2Fun:
        return logit

    @staticmethod
    def _fun(x: NDArray) -> NDArray:
        z = np.exp(-x)
        return 1 / (1 + z)

    @staticmethod
    def _dfun(x: NDArray) -> NDArray:
        z = np.exp(-x)
        return z / (1 + z)**2

    @staticmethod
    def _d2fun(x: NDArray) -> NDArray:
        z = np.exp(-x)
        return (z**2 - z) / (1 + z)**3


class Logit(C2Fun):
    """Logit function.

    """

    @property
    def inv(self) -> C2Fun:
        return expit

    @staticmethod
    def _fun(x: NDArray) -> NDArray:
        return np.log(x / (1 - x))

    @staticmethod
    def _dfun(x: NDArray) -> NDArray:
        x = np.asarray(x)
        return 1 / (x * (1 - x))

    @staticmethod
    def _d2fun(x: NDArray) -> NDArray:
        x = np.asarray(x)
        return (2*x - 1) / (x * (1 - x))**2


identity = Identity()
exp = Exp()
log = Log()
expit = Expit()
logit = Logit()


c2fun_dict: Dict[str, C2Fun] = {
    "identity": identity,
    "exp": exp,
    "log": log,
    "expit": expit,
    "logit": logit,
}
