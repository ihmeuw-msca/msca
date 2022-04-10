"""Continuously twice differentiable function. The instance of this class
provides the function and the derivative and the second derivative of the
function. It also contains the inverse of the function.

Example
-------
.. code-block:: python

    from msca.c2fun import exp

    x = [1, 2, 3]

    # exponential function
    y = exp(x)

    # derivative of the exponential function
    dy = exp(x, order=1)

    # second order derivative of the exponential function
    d2y = exp(x, order=2)

    # inverse function of the exponential function which is the log function
    z = exp.inv(x)

"""
from __future__ import annotations

from abc import ABC, abstractproperty, abstractstaticmethod
from typing import Dict

import numpy as np
from numpy.typing import NDArray


class C2Fun(ABC):
    """Abstract class that defines the interface for twice continuous function.
    It is callable and has an attribute :code:`inv` for the inverse function.

    """

    @abstractproperty
    def inv(self) -> C2Fun:
        """The inverse of the function such that :code:`x = fun.inv(fun(x))`.

        """
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
        """
        Parameters
        ----------
        x
            Provided independent variables.
        order
            Order of differentiation. This value has to be choose from 0, 1, or
            2. Default is 0.

        Returns
        -------
        NDArray
            The function, the derivative or the second derivative values.

        Raises
        ------
        ValueError
            Raised when the order is not 0, or 1, or 2.

        """
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

    .. math::

        f(x) = x, \quad
        f'(x) = 1, \quad
        f''(x) = 0


    """

    @property
    def inv(self) -> C2Fun:
        """The inverse of the identity function is the identity itself.

        """
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

    .. math::

        f(x) = \exp(x), \quad
        f'(x) = \exp(x), \quad
        f''(x) = \exp(x)


    """

    @property
    def inv(self) -> C2Fun:
        """The inverse of the exponential function is :class:`Log`.

        """
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

    .. math::

        f(x) = \log(x), \quad
        f'(x) = \\frac{1}{x}, \quad
        f''(x) = -\\frac{1}{x ^ 2}


    """

    @property
    def inv(self) -> C2Fun:
        """The inverse of the log function is the :class:`Exp`.

        """
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

    .. math::

        f(x) = \\frac{1}{1 + \exp(-x)}, \quad
        f'(x) = \\frac{\exp(-x)}{(1 + \exp(-x)) ^ 2}, \quad
        f''(x) = -\\frac{\exp(-2x) - \exp(-x)}{(1 + \exp(-x)) ^ 3}


    """

    @property
    def inv(self) -> C2Fun:
        """The inverse of the expit function is the :class:`Logit`.

        """
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

    .. math::

        f(x) = \log\\left(\\frac{x}{1 - x}\\right), \quad
        f'(x) = \\frac{1}{x(1 - x)}, \quad
        f''(x) = \\frac{2x - 1}{x ^ 2(1 - x) ^ 2}


    """

    @property
    def inv(self) -> C2Fun:
        """The inverse of the logit function is the :class:`Expit`.

        """
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


identity: C2Fun = Identity()
exp: C2Fun = Exp()
log: C2Fun = Log()
expit: C2Fun = Expit()
logit: C2Fun = Logit()

c2fun_dict: Dict[str, C2Fun] = {
    "identity": identity,
    "exp": exp,
    "log": log,
    "expit": expit,
    "logit": logit,
}
"""A dictionary that map function names with the function instances.

You can access the instances of :class:`C2Fun` through this dictionary.

.. code-block:: python

    from msca.c2fun import c2fun_dict

    exp = c2fun_dict['exp']

Or directly import the function.

.. code-block:: python

    from msca.c2fun import exp


"""
