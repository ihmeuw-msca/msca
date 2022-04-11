import numpy as np
from msca.linalg.matrix import asmatrix
from msca.optim.solver import NTSolver

np.random.seed(123)
mat = asmatrix(np.eye(5))
vec = np.random.randn(5)


def objective(x):
    r = vec - mat.dot(x)
    return 0.5*(r**2).sum()


def gradient(x):
    r = vec - mat.dot(x)
    return -mat.T.dot(r)


def hessian(x):
    return mat.T.dot(mat)


def test_ntsolver():
    solver = NTSolver(
        objective,
        gradient,
        hessian,
    )
    result = solver.minimize(
        x0=np.zeros(5), gtol=1e-10, xtol=0.0,
    )
    assert result.success
    assert np.allclose(result.x, vec)
