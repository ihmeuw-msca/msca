import numpy as np
from msca.linalg.matrix import asmatrix
from msca.optim.solver import IPSolver

np.random.seed(123)
mat = asmatrix(np.eye(5))
vec = np.random.randn(5)
cmat = asmatrix(np.eye(5))
cvec = np.zeros(5)


def objective(x):
    r = vec - mat.dot(x)
    return 0.5*(r**2).sum()


def gradient(x):
    r = vec - mat.dot(x)
    return -mat.T.dot(r)


def hessian(x):
    return mat.T.dot(mat)


def test_ipsolver():
    solver = IPSolver(
        objective,
        gradient,
        hessian,
        cmat,
        cvec
    )
    result = solver.minimize(
        x0=np.zeros(5), gtol=1e-10, xtol=0.0, mtol=1e-10, m_freq=1
    )
    assert result.success
    assert np.allclose(result.x, np.minimum(0, vec))
