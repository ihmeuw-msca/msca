from functools import partial

import numpy as np
import pytest
from msca.c2fun import c2fun_dict


def ad_dfun(fun, x, eps=1e-16):
    return fun(x + eps*1j).imag/eps


def pos_x():
    np.random.seed(123)
    return 0.1 + 0.8*np.random.rand(10)


def x():
    np.random.seed(123)
    return np.random.randn(10)


@pytest.mark.parametrize(("name", "x"),
                         [("identity", x()),
                          ("exp", x()),
                          ("log", pos_x()),
                          ("expit", x()),
                          ("logit", pos_x())])
def test_c2fun(name, x):
    fun = c2fun_dict[name]
    assert np.allclose(fun(x, order=1), ad_dfun(partial(fun, order=0), x))
    assert np.allclose(fun(x, order=2), ad_dfun(partial(fun, order=1), x))
    assert np.allclose(x, fun.inv(fun(x)))


@pytest.mark.parametrize(("name", "x"), [("logerfc", x())])
def test_logerfc(name, x):
    fun = c2fun_dict[name]
    assert np.allclose(fun(x, order=1), ad_dfun(partial(fun, order=0), x))
    assert np.allclose(fun(x, order=2), ad_dfun(partial(fun, order=1), x))
