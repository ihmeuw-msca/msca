from __future__ import annotations

from typing import Any, Protocol

import numpy
from numpy.typing import ArrayLike, NDArray
from scipy.sparse import csc_matrix, csr_matrix, spdiags
from scipy.sparse.linalg import spsolve


class Matrix(Protocol):

    def scale_rows(self, x: ArrayLike) -> Matrix:
        """Scale rows of the matrix.

        """

    def scale_cols(self, x: ArrayLike) -> Matrix:
        """Scale columns of the matrix.

        """

    def solve(self, x: ArrayLike) -> NDArray:
        """Solve the linear system.

        """

    def to_numpy(self) -> NDArray:
        """Convert to a numpy array.

        """


class NumpyMatrix(numpy.ndarray):

    def __new__(cls, *args, **kwargs):
        return numpy.asarray(*args, **kwargs).view(cls)

    def __init__(self, *args, **kwargs):
        if self.ndim != 2:
            raise ValueError("Matrix must be two dimensional.")

    def scale_rows(self, x: ArrayLike) -> NumpyMatrix:
        return numpy.asarray(x)[:, numpy.newaxis] * self

    def scale_cols(self, x: ArrayLike) -> NumpyMatrix:
        return self * numpy.asarray(x)

    def solve(self, x: ArrayLike) -> NDArray:
        return numpy.linalg.solve(self, x)

    def to_numpy(self) -> NDArray:
        return numpy.asarray(self.copy())

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={self.shape})"


class CSRMatrix(csr_matrix):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.ndim != 2:
            raise ValueError("Matrix must be two dimensional.")
        self._self = csr_matrix(self)

    @property
    def T(self) -> CSCMatrix:
        return CSCMatrix(self._self.T)

    def scale_rows(self, x: NDArray) -> CSRMatrix:
        x = numpy.asarray(x)
        return CSRMatrix(spdiags(x, 0, len(x), len(x)) * self)

    def scale_cols(self, x: NDArray) -> CSRMatrix:
        x = numpy.asarray(x)
        result = self.copy()
        result.data *= x[result.indices]
        return CSRMatrix(result)

    def solve(self, x: NDArray) -> NDArray:
        x = numpy.asarray(x)
        return spsolve(self, x)

    def to_numpy(self) -> NDArray:
        return self.toarray()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={self.shape})"


class CSCMatrix(csc_matrix):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.ndim != 2:
            raise ValueError("Matrix must be two dimensional.")
        self._self = csc_matrix(self)

    @property
    def T(self) -> CSRMatrix:
        return CSRMatrix(self._self.T)

    def scale_rows(self, x: NDArray) -> CSCMatrix:
        x = numpy.asarray(x)
        result = self.copy()
        result.data *= x[result.indices]
        return CSCMatrix(result)

    def scale_cols(self, x: NDArray) -> CSRMatrix:
        x = numpy.asarray(x)
        return CSCMatrix(self * spdiags(x, 0, len(x), len(x)))

    def solve(self, x: NDArray) -> NDArray:
        x = numpy.asarray(x)
        return spsolve(self, x)

    def to_numpy(self) -> NDArray:
        return self.toarray()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={self.shape})"


matrix_classes = (
    NumpyMatrix,
    CSCMatrix,
    CSRMatrix,
)


matrix_class_dict = {
    super(matrix_class): matrix_class
    for matrix_class in matrix_classes
}


def asmatrix(data: Any) -> Matrix:
    if isinstance(data, matrix_classes):
        return data
    if type(data) not in matrix_class_dict.keys():
        raise TypeError(f"Cannot convert {type(data)} to a matrix.")
    return matrix_class_dict[type(data)](data)
