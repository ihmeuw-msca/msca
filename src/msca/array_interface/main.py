from abc import ABC, abstractmethod
from typing import Literal

from pandas import DataFrame

from msca.typing import Array, DenseArray, JAXArray, NDArray, SPArray


class ArrayInterface(ABC):
    def __init__(self, dtype: Literal["float32", "float64"] = "float64") -> None:
        if dtype not in ["float32", "float64"]:
            raise ValueError("'dtype' must be either 'float32' or 'float64'")
        self.dtype = dtype

    def _validate_array(self, arr: Array) -> Array:
        if not isinstance(arr, (NDArray, JAXArray, SPArray)):
            raise TypeError(
                f"Input array must be numpy, jax, or sparse array, not {type(arr)}"
            )
        return arr

    def _validate_matrix(self, arr: Array, square: bool = False) -> Array:
        if arr.ndim != 2:
            raise ValueError("Matrix must have two dimensions")
        if square and (arr.shape[0] != arr.shape[1]):
            raise ValueError("Matrix must be square")
        return arr

    def _validate_vector(self, arr: Array) -> Array:
        if arr.ndim != 1:
            raise ValueError("Vector must have one dimension")
        return arr

    @abstractmethod
    def as_array(self, arr: Array) -> Array:
        """Convert an array to the desired type."""

    @abstractmethod
    def as_dense_array(self, arr: Array) -> DenseArray:
        """Convert an array to a dense array."""

    @abstractmethod
    def vstack(self, arrs: list[Array]) -> Array:
        """Stack arrays vertically."""

    @abstractmethod
    def vstack_dense_array(self, arrs: list[DenseArray]) -> DenseArray:
        """Stack dense arrays vertically."""

    @abstractmethod
    def hstack(self, arrs: list[Array]) -> Array:
        """Stack arrays horizontally."""

    @abstractmethod
    def hstack_dense_array(self, arrs: list[DenseArray]) -> DenseArray:
        """Stack dense arrays horizontally."""

    @abstractmethod
    def block_diag(self, arrs: list[Array]) -> Array:
        """Stack arrays in a block diagonal fashion."""

    @abstractmethod
    def identity(self, size: int) -> Array:
        """Create an identity array."""

    @abstractmethod
    def empty(self, shape: tuple[int, ...]) -> Array:
        """Create an empty array."""

    @abstractmethod
    def dataframe_to_array(self, df: DataFrame) -> Array:
        """Convert a DataFrame to an array."""

    @abstractmethod
    def scale_rows(self, mat: Array, vec: Array) -> Array:
        """Scale rows of a matrix."""

    @abstractmethod
    def scale_cols(self, mat: Array, vec: Array) -> Array:
        """Scale columns of a matrix."""

    @abstractmethod
    def solve(self, mat: Array, vec: Array, method: str = "", **kwargs) -> Array:
        """Solve a linear system."""
