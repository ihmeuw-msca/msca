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

from abc import ABC, abstractmethod

from msca.typing import DenseArray


class C2Fun(ABC):
    """Abstract class that defines the interface for twice continuous function.
    To inherit this class, user much provide :meth:`fun`, :meth:`dfun` and
    :meth:`d2fun`. And if inverse of the function is defined and implemented
    please override :attr:`inv`, otherwise please raise
    :code:`NotImplementedError`.

    The instance of :class:`C2Fun` is callable, with the signature defined in
    the :meth:`__call__` function.

    """

    @property
    @abstractmethod
    def inv(self) -> C2Fun:
        """The inverse of the function such that :code:`x = fun.inv(fun(x))`."""
        pass

    @staticmethod
    @abstractmethod
    def fun(x: DenseArray) -> DenseArray:
        """Implementation of the function.

        Parameters
        ----------
        x
            Provided independent variable.

        """
        pass

    @staticmethod
    @abstractmethod
    def dfun(x: DenseArray) -> DenseArray:
        """Implementation of the derivative of the function.

        Parameters
        ----------
        x
            Provided independent variable.

        """
        pass

    @staticmethod
    @abstractmethod
    def d2fun(x: DenseArray) -> DenseArray:
        """Implementation of the second order derivative of the function.

        Parameters
        ----------
        x
            Provided independent variable.

        """
        pass

    def __call__(self, x: DenseArray, order: int = 0) -> DenseArray:
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
        DenseArray
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
