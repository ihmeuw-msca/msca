import numpy as np
import pytest
from msca.linalg.matrix import Matrix, matrix_classes


@pytest.fixture
def mat():
    return np.array([[1.0, 2.0], [3.0, 4.0]])


@pytest.mark.parametrize("x", [np.ones(2), np.ones((2, 2))])
@pytest.mark.parametrize("matrix_class", matrix_classes)
def test_dot(mat, x, matrix_class):
    mat = matrix_class(mat)
    result = mat.dot(x)
    assert np.allclose(result, mat.to_numpy().dot(x))
    if x.ndim == 2:
        assert isinstance(result, Matrix)
