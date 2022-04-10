from functools import partial

import numpy as np
import pytest
from msca.c2fun import c2fun_dict


def ad_dfun(fun, x, eps=1e-16):
    return fun(x + eps*1j).imag/eps


@pytest.fixture
def x():
    np.random.seed(123)
    return 0.1 + 0.8*np.random.rand(3)


@pytest.mark.parametrize("name",
                         ["identity",
                          "exp",
                          "log",
                          "expit",
                          "logit"])
def test_c2fun(name, x):
    fun = c2fun_dict[name]
    assert np.allclose(fun(x, order=1), ad_dfun(partial(fun, order=0), x))
    assert np.allclose(fun(x, order=2), ad_dfun(partial(fun, order=1), x))
    assert np.allclose(x, fun.inv(fun(x)))
