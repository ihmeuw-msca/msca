import numpy as np
from msca.array_interface import NumpyArrayInterface
from msca.optim.solver import IPSolver

np.random.seed(123)
arrif = NumpyArrayInterface("float32")
mat = arrif.as_array(np.eye(5))
vec = arrif.as_array(np.random.randn(5))
cmat = arrif.as_array(np.eye(5))
cvec = arrif.as_array(np.zeros(5))


def objective(x):
    r = vec - mat.dot(x)
    return 0.5 * (r**2).sum()


def gradient(x):
    r = vec - mat.dot(x)
    return -mat.T.dot(r)


def hessian(x):
    return mat.T.dot(mat)


def test_ipsolver():
    solver = IPSolver(objective, gradient, hessian, cmat, cvec, arrif)
    result = solver.minimize(
        x0=arrif.as_array(np.zeros(5)), gtol=1e-10, xtol=0.0, mtol=1e-10, m_freq=1
    )
    assert result.success
    assert np.allclose(result.x, np.minimum(0, vec))
