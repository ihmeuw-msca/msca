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

Note
----
All the concrete classes listed below already have module-level instance created
in this module. We suggest user to directly use these instances rather than
create new instances from the class. You can also access the instances from the
model-level variable :data:`c2fun_dict`.

"""
from __future__ import annotations

from abc import ABC, abstractproperty, abstractstaticmethod
from typing import Dict

import numpy as np
from numpy.typing import NDArray


class C2Fun(ABC):
    """Abstract class that defines the interface for twice continuous function.
    To inherit this class, user much provide :meth:`fun`, :meth:`dfun` and
    :meth:`d2fun`. And if inverse of the function is defined and implemented
    please override :attr:`inv`, otherwise please raise
    :code:`NotImplementedError`.

    The instance of :class:`C2Fun` is callable, with the signature defined in
    the :meth:`__call__` function.

    """

    @abstractproperty
    def inv(self) -> C2Fun:
        """The inverse of the function such that :code:`x = fun.inv(fun(x))`.

        """
        pass

    @abstractstaticmethod
    def fun(x: NDArray) -> NDArray:
        """Implementation of the function.

        Parameters
        ----------
        x
            Provided independent variable.

        """
        pass

    @abstractstaticmethod
    def dfun(x: NDArray) -> NDArray:
        """Implementation of the derivative of the function.

        Parameters
        ----------
        x
            Provided independent variable.

        """
        pass

    @abstractstaticmethod
    def d2fun(x: NDArray) -> NDArray:
        """Implementation of the second order derivative of the function.

        Parameters
        ----------
        x
            Provided independent variable.

        """
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
            return self.fun(x)
        if order == 1:
            return self.dfun(x)
        if order == 2:
            return self.d2fun(x)
        raise ValueError("Order has to be selected from 0, 1 or 2.")

    def __repr__(self) -> str:
        return f"{type(self).__name__}()"


class Identity(C2Fun):

    @property
    def inv(self) -> C2Fun:
        """The inverse of the identity function is the identity itself.

        """
        return self

    @staticmethod
    def fun(x: NDArray) -> NDArray:
        """
        .. math::

            f(x) = x

        Parameters
        ----------
        x
            Provided independent variable.

        """
        x = np.asarray(x)
        return x

    @staticmethod
    def dfun(x: NDArray) -> NDArray:
        """
        .. math::

            f'(x) = 1

        Parameters
        ----------
        x
            Provided independent variable.

        """
        x = np.asarray(x)
        return np.ones(x.size, dtype=x.dtype)

    @staticmethod
    def d2fun(x: NDArray) -> NDArray:
        """
        .. math::

            f''(x) = 0

        Parameters
        ----------
        x
            Provided independent variable.

        """
        x = np.asarray(x)
        return np.zeros(x.size, dtype=x.dtype)


class Exp(C2Fun):

    @property
    def inv(self) -> C2Fun:
        """The inverse of the exponential function is :class:`Log`.

        """
        return log

    @staticmethod
    def fun(x: NDArray) -> NDArray:
        """
        .. math::

            f(x) = \\exp(x)

        Parameters
        ----------
        x
            Provided independent variable.

        """
        return np.exp(x)

    @staticmethod
    def dfun(x: NDArray) -> NDArray:
        """
        .. math::

            f'(x) = \\exp(x)

        Parameters
        ----------
        x
            Provided independent variable.

        """
        return np.exp(x)

    @staticmethod
    def d2fun(x: NDArray) -> NDArray:
        """
        .. math::

            f''(x) = \\exp(x)

        Parameters
        ----------
        x
            Provided independent variable.

        """
        return np.exp(x)


class Log(C2Fun):

    @property
    def inv(self) -> C2Fun:
        """The inverse of the log function is the :class:`Exp`.

        """
        return exp

    @staticmethod
    def fun(x: NDArray) -> NDArray:
        """
        .. math::

            f(x) = \\log(x)

        Parameters
        ----------
        x
            Provided independent variable.

        """
        return np.log(x)

    @staticmethod
    def dfun(x: NDArray) -> NDArray:
        """
        .. math::

            f'(x) = \\frac{1}{x}

        Parameters
        ----------
        x
            Provided independent variable.

        """
        x = np.asarray(x)
        return 1 / x

    @staticmethod
    def d2fun(x: NDArray) -> NDArray:
        """
        .. math::

            f''(x) = -\\frac{1}{x^2}

        Parameters
        ----------
        x
            Provided independent variable.

        """
        x = np.asarray(x)
        return -1 / x**2


class Expit(C2Fun):
    """
    Note
    ----
    We use the form with :math:`\\exp(-x)` when :math:`x > 0`, and the form
    with :math:`\\exp(x)` when :math:`x \\le 0`.

    """

    @property
    def inv(self) -> C2Fun:
        """The inverse of the expit function is the :class:`Logit`.

        """
        return logit

    @staticmethod
    def fun(x: NDArray) -> NDArray:
        """
        .. math::
            f(x) = \\frac{1}{1 + \\exp(-x)} = \\frac{\\exp(x)}{1 + \\exp(x)}

        Parameters
        ----------
        x
            Provided independent variable.

        """
        x = np.asarray(x)
        y = np.zeros(x.size, dtype=x.dtype)

        pos_indices = x > 0
        z = np.exp(-x[pos_indices])
        y[pos_indices] = 1 / (1 + z)

        neg_indices = ~pos_indices
        z = np.exp(x[neg_indices])
        y[neg_indices] = z / (1 + z)

        return y

    @staticmethod
    def dfun(x: NDArray) -> NDArray:
        """
        .. math::
            f'(x) = \\frac{\\exp(-x)}{(1 + \\exp(-x)) ^ 2}
            = \\frac{\\exp(x)}{(1 + \\exp(x))^2}

        Parameters
        ----------
        x
            Provided independent variable.

        """
        x = np.asarray(x)
        y = np.zeros(x.size, dtype=x.dtype)

        pos_indices = x > 0
        z = np.exp(-x[pos_indices])
        y[pos_indices] = z / (1 + z)**2

        neg_indices = ~pos_indices
        z = np.exp(x[neg_indices])
        y[neg_indices] = z / (1 + z)**2

        return y

    @staticmethod
    def d2fun(x: NDArray) -> NDArray:
        """
        .. math::
            f''(x) = \\frac{\\exp(-2x) - \\exp(-x)}{(1 + \\exp(-x)) ^ 3}
            = \\frac{\\exp(x) - \\exp(2x)}{(1 + \\exp(x))^3}

        Parameters
        ----------
        x
            Provided independent variable.

        """
        x = np.asarray(x)
        y = np.zeros(x.size, dtype=x.dtype)

        pos_indices = x > 0
        z = np.exp(-x[pos_indices])
        y[pos_indices] = (z**2 - z) / (1 + z)**3

        neg_indices = ~pos_indices
        z = np.exp(x[neg_indices])
        y[neg_indices] = (z - z**2) / (1 + z)**3

        return y


class Logit(C2Fun):

    @property
    def inv(self) -> C2Fun:
        """The inverse of the logit function is the :class:`Expit`.

        """
        return expit

    @staticmethod
    def fun(x: NDArray) -> NDArray:
        """
        .. math::

            f(x) = \\log\\left(\\frac{x}{1 - x}\\right)

        Parameters
        ----------
        x
            Provided independent variable.

        """
        return np.log(x / (1 - x))

    @staticmethod
    def dfun(x: NDArray) -> NDArray:
        """
        .. math::

            f'(x) = \\frac{1}{x(1 - x)}

        Parameters
        ----------
        x
            Provided independent variable.

        """
        x = np.asarray(x)
        return 1 / (x * (1 - x))

    @staticmethod
    def d2fun(x: NDArray) -> NDArray:
        """
        .. math::

            f''(x) = \\frac{2x - 1}{x ^ 2(1 - x) ^ 2}

        Parameters
        ----------
        x
            Provided independent variable.

        """
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


"""
