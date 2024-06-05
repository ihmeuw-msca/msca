import numpy as np
import pytest
from msca.linalg.matrix import Matrix, asmatrix, matrix_classes
from scipy.sparse import csc_matrix, csr_matrix


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


@pytest.mark.parametrize("x", [np.array([1.0, 2.0])])
@pytest.mark.parametrize("matrix_class", matrix_classes)
def test_scale_rows(mat, x, matrix_class):
    mat = matrix_class(mat)
    result = mat.scale_rows(x)
    assert isinstance(result, Matrix)
    assert np.allclose(result.to_numpy(), mat.to_numpy() * x[:, np.newaxis])


@pytest.mark.parametrize("x", [np.array([1.0, 2.0])])
@pytest.mark.parametrize("matrix_class", matrix_classes)
def test_scale_cols(mat, x, matrix_class):
    mat = matrix_class(mat)
    result = mat.scale_cols(x)
    assert isinstance(result, Matrix)
    assert np.allclose(result.to_numpy(), mat.to_numpy() * x)


@pytest.mark.parametrize("x", [np.array([1.0, 2.0])])
@pytest.mark.parametrize("matrix_class", matrix_classes)
def test_solve(mat, x, matrix_class):
    mat = matrix_class(mat)
    result = mat.solve(x)
    assert np.allclose(x, mat.dot(result))


@pytest.mark.parametrize("matrix_class", matrix_classes)
def test_to_numpy(mat, matrix_class):
    mat = matrix_class(mat)
    assert isinstance(mat.to_numpy(), np.ndarray)


@pytest.mark.parametrize(
    "mat", [np.eye(2), csc_matrix(np.eye(2)), csr_matrix(np.eye(2))]
)
def test_asmatrix(mat):
    mat = asmatrix(mat)
    assert isinstance(mat, Matrix)
