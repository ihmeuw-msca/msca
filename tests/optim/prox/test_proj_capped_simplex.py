import numpy as np
import pytest
from msca.optim.prox import proj_capped_simplex

s = 2.0


def case0():
    x = np.array([1.0, 1.0, 1.0, 1.0])
    y = np.array([0.5, 0.5, 0.5, 0.5])
    return x, y


def case1():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    return x, y


def case2():
    x = np.array([2.0, 2.0, 3.0, 3.0])
    y = np.array([0.0, 0.0, 1.0, 1.0])
    return x, y


@pytest.mark.parametrize(("x", "y"), [case0(), case1(), case2()])
def test_proj_capped_simplex(x, y):
    assert np.allclose(proj_capped_simplex(x, s), y)
