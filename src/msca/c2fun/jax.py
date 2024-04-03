import jax.numpy as jnp
from jax.scipy.special import erfc

from msca.c2fun.main import C2Fun
from msca.typing import JAXArray


class Identity(C2Fun):
    @property
    def inv(self) -> C2Fun:
        """The inverse of the identity function is the identity itself."""
        return self

    @staticmethod
    def fun(x: JAXArray) -> JAXArray:
        """
        .. math::

            f(x) = x

        Parameters
        ----------
        x
            Provided independent variable.

        """
        return x

    @staticmethod
    def dfun(x: JAXArray) -> JAXArray:
        """
        .. math::

            f'(x) = 1

        Parameters
        ----------
        x
            Provided independent variable.

        """
        return jnp.ones(x.shape, dtype=x.dtype)

    @staticmethod
    def d2fun(x: JAXArray) -> JAXArray:
        """
        .. math::

            f''(x) = 0

        Parameters
        ----------
        x
            Provided independent variable.

        """
        return jnp.zeros(x.shape, dtype=x.dtype)


class Exp(C2Fun):
    @property
    def inv(self) -> C2Fun:
        """The inverse of the exponential function is :class:`Log`."""
        return log

    @staticmethod
    def fun(x: JAXArray) -> JAXArray:
        """
        .. math::

            f(x) = \\exp(x)

        Parameters
        ----------
        x
            Provided independent variable.

        """
        return jnp.exp(x)

    @staticmethod
    def dfun(x: JAXArray) -> JAXArray:
        """
        .. math::

            f'(x) = \\exp(x)

        Parameters
        ----------
        x
            Provided independent variable.

        """
        return jnp.exp(x)

    @staticmethod
    def d2fun(x: JAXArray) -> JAXArray:
        """
        .. math::

            f''(x) = \\exp(x)

        Parameters
        ----------
        x
            Provided independent variable.

        """
        return jnp.exp(x)


class Log(C2Fun):
    @property
    def inv(self) -> C2Fun:
        """The inverse of the log function is the :class:`Exp`."""
        return exp

    @staticmethod
    def fun(x: JAXArray) -> JAXArray:
        """
        .. math::

            f(x) = \\log(x)

        Parameters
        ----------
        x
            Provided independent variable.

        """
        return jnp.log(x)

    @staticmethod
    def dfun(x: JAXArray) -> JAXArray:
        """
        .. math::

            f'(x) = \\frac{1}{x}

        Parameters
        ----------
        x
            Provided independent variable.

        """
        return 1 / x

    @staticmethod
    def d2fun(x: JAXArray) -> JAXArray:
        """
        .. math::

            f''(x) = -\\frac{1}{x^2}

        Parameters
        ----------
        x
            Provided independent variable.

        """
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
        """The inverse of the expit function is the :class:`Logit`."""
        return logit

    @staticmethod
    def fun(x: JAXArray) -> JAXArray:
        """
        .. math::
            f(x) = \\frac{1}{1 + \\exp(-x)} = \\frac{\\exp(x)}{1 + \\exp(x)}

        Parameters
        ----------
        x
            Provided independent variable.

        """
        z = jnp.where(x > 0, jnp.exp(-x), jnp.exp(x))
        y = jnp.where(x > 0, 1, z) / (1 + z)
        return y

    @staticmethod
    def dfun(x: JAXArray) -> JAXArray:
        """
        .. math::
            f'(x) = \\frac{\\exp(-x)}{(1 + \\exp(-x)) ^ 2}
            = \\frac{\\exp(x)}{(1 + \\exp(x))^2}

        Parameters
        ----------
        x
            Provided independent variable.

        """
        z = jnp.where(x > 0, jnp.exp(-x), jnp.exp(x))
        y = z / (1 + z) ** 2
        return y

    @staticmethod
    def d2fun(x: JAXArray) -> JAXArray:
        """
        .. math::
            f''(x) = \\frac{\\exp(-2x) - \\exp(-x)}{(1 + \\exp(-x)) ^ 3}
            = \\frac{\\exp(x) - \\exp(2x)}{(1 + \\exp(x))^3}

        Parameters
        ----------
        x
            Provided independent variable.

        """
        z = jnp.where(x > 0, jnp.exp(-x), jnp.exp(x))
        y = jnp.where(x > 0, z**2 - z, z - z**2) / (1 + z) ** 3
        return y


class Logit(C2Fun):
    @property
    def inv(self) -> C2Fun:
        """The inverse of the logit function is the :class:`Expit`."""
        return expit

    @staticmethod
    def fun(x: JAXArray) -> JAXArray:
        """
        .. math::

            f(x) = \\log\\left(\\frac{x}{1 - x}\\right)

        Parameters
        ----------
        x
            Provided independent variable.

        """
        return jnp.log(x / (1 - x))

    @staticmethod
    def dfun(x: JAXArray) -> JAXArray:
        """
        .. math::

            f'(x) = \\frac{1}{x(1 - x)}

        Parameters
        ----------
        x
            Provided independent variable.

        """
        return 1 / (x * (1 - x))

    @staticmethod
    def d2fun(x: JAXArray) -> JAXArray:
        """
        .. math::

            f''(x) = \\frac{2x - 1}{x ^ 2(1 - x) ^ 2}

        Parameters
        ----------
        x
            Provided independent variable.

        """
        return (2 * x - 1) / (x * (1 - x)) ** 2


class Logerfc(C2Fun):
    """Logerfc function is a special function that appears in stochastic
    frontier analysis. Erfc function can be written as

    .. math::
        \\mathrm{erfc}(x) =
        1 - \\frac{2}{\\sqrt(\\pi)}\\int_0^x \\exp(-t^2) \\,dt

    Note
    ----
    Erfc function converges to zero very fast when :math:`x` increases, and it
    will cause numerical issue when compute Logerfc. We use asymptotic
    approximation to avoid this problem.

    """

    @property
    def inv(self) -> C2Fun:
        """The inverse of the logerfc function is not implemeneted.

        Raises
        ------
        NotImplementedError
            Raised whenever using this property.

        """
        raise NotImplementedError

    @staticmethod
    def fun(x: JAXArray) -> JAXArray:
        """
        .. math::

            f(x) = \\log(\\mathrm{erfc}(x))

        When :math:`x \\ge 25` we use approximation.

        .. math::

            f(x) \\approx -x^2 + \\log\\left(1 - \\frac{1}{2 x^2}\\right) -
            \\log(\\sqrt{\\pi} x)

        Parameters
        ----------
        x
            Provided independent variable.

        """
        y = jnp.where(
            x < 25,
            jnp.log(erfc(x)),
            -(x**2) + jnp.log(1 - 0.5 / x**2) - jnp.log(jnp.sqrt(jnp.pi) * x),
        )
        return y

    @staticmethod
    def dfun(x: JAXArray) -> JAXArray:
        """
        .. math::

            f'(x) = -\\frac{2\\exp(-x^2)}{\\sqrt{\\pi}\\mathrm{erfc}(x)}

        When :math:`x \\ge 25` we use approximation.

        .. math::

            f(x) \\approx -2 x - \\frac{1}{x} + \\frac{2}{2 x^3 - x}

        Parameters
        ----------
        x
            Provided independent variable.

        """
        y = jnp.where(
            x < 25,
            -2 * jnp.exp(-(x**2)) / (erfc(x) * jnp.sqrt(jnp.pi)),
            -2 * x - 1 / x + 2 / (2 * x**3 - x),
        )
        return y

    @staticmethod
    def d2fun(x: JAXArray) -> JAXArray:
        """
        .. math::

            f''(x) = \\frac{4 x \\exp(-x^2)}{\\sqrt{\\pi} \\mathrm{erfc}(x)} -
            \\frac{4 \\exp(-2 x^2)}{\\pi \\mathrm{erfc}(x)^2} =
            -2 x f'(x) - f'(x)^2

        Parameters
        ----------
        x
            Provided independent variable.

        """
        d = Logerfc.dfun(x)
        return -2 * x * d - d**2


identity: C2Fun = Identity()
exp: C2Fun = Exp()
log: C2Fun = Log()
expit: C2Fun = Expit()
logit: C2Fun = Logit()
logerfc: C2Fun = Logerfc()

c2fun_dict: dict[str, C2Fun] = {
    "identity": identity,
    "exp": exp,
    "log": log,
    "expit": expit,
    "logit": logit,
    "logerfc": logerfc,
}
"""A dictionary that map function names with the function instances.

You can access the instances of :class:`C2Fun` through this dictionary.

.. code-block:: python

    from msca.c2fun.jax import c2fun_dict

    exp = c2fun_dict['exp']


"""
