import numpy as np
import scipy as sp
from pandas import DataFrame

from msca.array_interface.main import ArrayInterface
from msca.typing import Array, NDArray, SPArray


class NumpyArrayInterface(ArrayInterface):
    def as_array(self, arr: Array) -> NDArray:
        arr = self._validate_array(arr)
        if isinstance(arr, SPArray):
            arr = arr.toarray()
        return np.asarray(arr, dtype=self.dtype)

    def as_dense_array(self, arr: Array) -> NDArray:
        return self.as_array(arr)

    def vstack(self, arrs: list[Array]) -> NDArray:
        return np.vstack([self.as_array(arr) for arr in arrs])

    def vstack_dense_array(self, arrs: list[Array]) -> NDArray:
        return self.vstack(arrs)

    def hstack(self, arrs: list[Array]) -> NDArray:
        return np.hstack([self.as_array(arr) for arr in arrs])

    def hstack_dense_array(self, arrs: list[Array]) -> NDArray:
        return self.hstack(arrs)

    def block_diag(self, arrs: list[Array]) -> NDArray:
        return sp.linalg.block_diag(*[self.as_array(arr) for arr in arrs])

    def identity(self, size: int) -> NDArray:
        return np.identity(size, dtype=self.dtype)

    def empty(self, shape: tuple[int, ...]) -> NDArray:
        return np.empty(shape, dtype=self.dtype)

    def dataframe_to_array(self, df: DataFrame) -> NDArray:
        return self.as_array(df.to_numpy())

    def scale_rows(self, mat: NDArray, vec: NDArray) -> NDArray:
        mat = self._validate_matrix(self.as_array(mat))
        vec = self._validate_vector(self.as_dense_array(vec))
        return vec[:, np.newaxis] * mat

    def scale_cols(self, mat: NDArray, vec: NDArray) -> NDArray:
        mat = self._validate_matrix(self.as_array(mat))
        vec = self._validate_vector(self.as_dense_array(vec))
        return mat * vec

    def solve(self, mat: NDArray, vec: NDArray, method="", **kwargs) -> NDArray:
        mat = self._validate_matrix(self.as_array(mat), square=True)
        vec = self._validate_vector(self.as_dense_array(vec))
        if method == "":
            soln = np.linalg.solve(mat, vec, **kwargs)
        elif method == "cg":
            soln, info = sp.sparse.linalg.cg(mat, vec, **kwargs)
            if info > 0:
                raise RuntimeError(f"CG convergence not achieved. with {info=:}")
        else:
            raise ValueError(f"{method=:} is not supported.")
        return soln
