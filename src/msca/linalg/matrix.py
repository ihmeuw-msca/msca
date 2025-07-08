from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Type

import numpy as np
import scipy as sp
from numpy.typing import ArrayLike, NDArray


class Matrix(ABC):
    """A matrix wrapper class that unifies the interfaces for different types
    of array.

    Raises
    ------
    ValueError
        Raised when the number of dimension is not two.

    """

    def __init__(self):
        if self.ndim != 2:
            raise ValueError("Matrix must be two dimensional.")

    @abstractmethod
    def scale_rows(self, x: ArrayLike) -> Matrix:
        """Scale rows of the matrix.

        Parameters
        ----------
        x
            A vector that contains scalings for rows of the matrix. The size of
            the vector should align with the number of rows of the matrix.

        Returns
        -------
        Matrix
            The scaled matrix.

        """
        pass

    @abstractmethod
    def scale_cols(self, x: ArrayLike) -> Matrix:
        """Scale columns of the matrix.

        Parameters
        ----------
        x
            A vector that contains scalings for columns of the matrix. The size
            of the vector should align with the number of columns of the matrix.

        Returns
        -------
        Matrix
            The scaled matrix.

        """
        pass

    @abstractmethod
    def solve(self, x: ArrayLike) -> NDArray:
        """Solve the linear system.

        Parameters
        ----------
        x
            A vector that represent the right hand side of the linear equation.
            The size of the vector should align with the number of rows of the
            matrix.

        Returns
        -------
        NDArray
            The solution of the linear system.

        """
        pass

    @abstractmethod
    def to_numpy(self) -> NDArray:
        """Convert to a numpy array.

        Returns
        -------
        NDArray
            A numpy representation of the matrix.

        """
        pass


class NumpyMatrix(np.ndarray, Matrix):
    def __new__(cls, *args, **kwargs):
        return np.asarray(*args, **kwargs).view(cls)

    def __init__(self, *args, **kwargs):
        super(Matrix, self).__init__()

    def dot(self, x: ArrayLike) -> NDArray | Matrix:
        """Dot product operator for the matrix.

        Parameters
        ----------
        x
            Give vector or matrix.

        Returns
        -------
        NDArray | Matrix
            Results of the dot product. If the result is two dimensional it will
            be an instance of the :py:class:`Matrix` class.

        """
        result = np.dot(self, x)
        if result.ndim == 1:
            return np.asarray(result)
        return result

    def scale_rows(self, x: ArrayLike) -> NumpyMatrix:
        return np.asarray(x)[:, np.newaxis] * self

    def scale_cols(self, x: ArrayLike) -> NumpyMatrix:
        return self * np.asarray(x)

    def solve(self, x: ArrayLike, method: str = "", **kwargs) -> NDArray:
        x = np.asarray(x)
        if method == "":
            result = np.linalg.solve(self, x, **kwargs)
        elif method == "cg":
            result, info = sp.linalg.cg(self, x, **kwargs)
            if info > 0:
                raise RuntimeError(
                    f"CG convergence not achieved. with {info=:}"
                )
        else:
            raise ValueError(f"{method=:} is not supported.")
        return result

    def to_numpy(self) -> NDArray:
        return np.asarray(self.copy())

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={self.shape})"


class CSRMatrix(sp.sparse.csr_matrix, Matrix):
    def __init__(self, *args, **kwargs):
        super(sp.sparse.csr_matrix, self).__init__(*args, **kwargs)
        super(Matrix, self).__init__()

    @property
    def T(self) -> CSCMatrix:
        """Transpose of the matrix. It will be an instance of the
        :py:class:`CSCMatrix`.

        """
        return CSCMatrix(self.transpose())

    def dot(self, x: ArrayLike) -> NDArray | Matrix:
        """Dot product operator for the matrix.

        Parameters
        ----------
        x
            Give vector or matrix.

        Returns
        -------
        NDArray | Matrix
            Results of the dot product. If the result is two dimensional it will
            be an instance of the :py:class:`Matrix` class.

        """
        result = super(sp.sparse.csr_matrix, self).dot(x)
        if result.ndim == 1:
            return result
        return asmatrix(result)

    def scale_rows(self, x: NDArray) -> CSRMatrix:
        x = np.asarray(x)
        return CSRMatrix(sp.sparse.spdiags(x, 0, len(x), len(x)) * self)

    def scale_cols(self, x: NDArray) -> CSRMatrix:
        x = np.asarray(x)
        result = self.copy()
        result.data *= x[result.indices]
        return CSRMatrix(result)

    def solve(self, x: NDArray, method: str = "", **kwargs) -> NDArray:
        x = np.asarray(x)
        if method == "":
            result = sp.sparse.linalg.spsolve(self, x, **kwargs)
        elif method == "cg":
            result, info = sp.sparse.linalg.cg(self, x, **kwargs)
            if info > 0:
                raise RuntimeError(
                    f"CG convergence not achieved. with {info=:}"
                )
        else:
            raise ValueError(f"{method=:} is not supported.")
        return result

    def to_numpy(self) -> NDArray:
        return self.toarray()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={self.shape})"


class CSCMatrix(sp.sparse.csc_matrix, Matrix):
    def __init__(self, *args, **kwargs):
        super(sp.sparse.csc_matrix, self).__init__(*args, **kwargs)
        super(Matrix, self).__init__()

    @property
    def T(self) -> CSRMatrix:
        """Transpose of the matrix. It will be an instance of the
        :py:class:`CSRMatrix`.

        """
        return CSRMatrix(self.transpose())

    def dot(self, x: ArrayLike) -> Matrix:
        """Dot product operator for the matrix.

        Parameters
        ----------
        x
            Give vector or matrix.

        Returns
        -------
        NDArray | Matrix
            Results of the dot product. If the result is two dimensional it will
            be an instance of the :py:class:`Matrix` class.

        """
        result = super(sp.sparse.csc_matrix, self).dot(x)
        if result.ndim == 1:
            return result
        return asmatrix(result)

    def scale_rows(self, x: NDArray) -> CSCMatrix:
        x = np.asarray(x)
        result = self.copy()
        result.data *= x[result.indices]
        return CSCMatrix(result)

    def scale_cols(self, x: NDArray) -> CSRMatrix:
        x = np.asarray(x)
        return CSCMatrix(self * sp.sparse.spdiags(x, 0, len(x), len(x)))

    def solve(self, x: NDArray, method: str = "", **kwargs) -> NDArray:
        x = np.asarray(x)
        if method == "":
            result = sp.sparse.linalg.spsolve(self, x, **kwargs)
        elif method == "cg":
            result, info = sp.sparse.linalg.cg(self, x, **kwargs)
            if info > 0:
                raise RuntimeError(
                    f"CG convergence not achieved. with {info=:}"
                )
        else:
            raise ValueError(f"{method=:} is not supported.")
        return result

    def to_numpy(self) -> NDArray:
        return self.toarray()

    def __repr__(self) -> str:
        return f"{type(self).__name__}(shape={self.shape})"


matrix_classes: tuple[Type, ...] = (
    NumpyMatrix,
    CSCMatrix,
    CSRMatrix,
)
"""A collection of all matrix classes.

:meta hide-value:

"""

matrix_class_dict: dict[Type, Type] = {
    matrix_class.__base__: matrix_class for matrix_class in matrix_classes
}
"""Matrix classes organize in a dictionary, with key as their parent class and
value as the matrix class.

:meta hide-value:

"""


def asmatrix(data: Any) -> Matrix:
    """Convert data into an instance of the matrix class based on its type.

    Parameters
    ----------
    data
        Given data, it should be compatible with the matrix classes. If data is
        an instance of :py:class:`Matrix` already, it will be returned as it is.

    Raises
    ------
    TypeError
        Raised when the type of the data is not compatible with the matrix
        classes.

    Returns
    -------
    Matrix
        The converted matrix.

    """
    if isinstance(data, matrix_classes):
        return data
    if type(data) not in matrix_class_dict.keys():
        raise TypeError(f"Cannot convert {type(data)} to a matrix.")
    return matrix_class_dict[type(data)](data)
