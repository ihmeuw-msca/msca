import jax
import jax.numpy as jnp
import pytest
from msca.c2fun.jax import c2fun_dict


def pos_x():
    key = jax.random.key(123)
    return 0.1 + 0.8 * jax.random.uniform(key, shape=(10,))


def x():
    key = jax.random.key(123)
    return jax.random.normal(key, shape=(10,))


@pytest.mark.parametrize(
    ("name", "x"),
    [
        ("identity", x()),
        ("exp", x()),
        ("log", pos_x()),
        ("expit", x()),
        ("logit", pos_x()),
    ],
)
def test_c2fun(name, x):
    fun = c2fun_dict[name]
    dfun = jax.jit(jax.jacobian(fun.fun))
    d2fun = jax.jit(jax.jacobian(fun.dfun))

    assert jnp.allclose(fun(x, order=1), jax.numpy.diag(dfun(x)))
    assert jnp.allclose(fun(x, order=2), jax.numpy.diag(d2fun(x)))
    assert jnp.allclose(x, fun.inv(fun(x)))


@pytest.mark.parametrize(("name", "x"), [("logerfc", x())])
def test_logerfc(name, x):
    fun = c2fun_dict[name]
    dfun = jax.jit(jax.jacobian(fun.fun))
    d2fun = jax.jit(jax.jacobian(fun.dfun))

    assert jnp.allclose(fun(x, order=1), jax.numpy.diag(dfun(x)))
    assert jnp.allclose(fun(x, order=2), jax.numpy.diag(d2fun(x)))
