import numpy as np
import pandas as pd
import scipy.sparse as sps
from pandas import DataFrame

from msca.array_interface.main import ArrayInterface
from msca.typing import Array, NDArray, SPArray


class SparseArrayInterface(ArrayInterface):
    def as_array(self, arr: Array) -> SPArray:
        arr = self._validate_array(arr)
        return sps.csc_array(arr, dtype=self.dtype)

    def as_dense_array(self, arr: Array) -> NDArray:
        arr = self._validate_array(arr)
        if isinstance(arr, SPArray):
            arr = arr.toarray()
        return np.asarray(arr, dtype=self.dtype)

    def vstack(self, arrs: list[Array]) -> SPArray:
        return sps.vstack([self.as_array(arr) for arr in arrs], format="csc")

    def vstack_dense_array(self, arrs: list[Array]) -> NDArray:
        return np.vstack([self.as_dense_array(arr) for arr in arrs])

    def hstack(self, arrs: list[Array]) -> SPArray:
        return sps.hstack([self.as_array(arr) for arr in arrs], format="csc")

    def hstack_dense_array(self, arrs: list[Array]) -> NDArray:
        return np.hstack([self.as_dense_array(arr) for arr in arrs])

    def block_diag(self, arrs: list[Array]) -> SPArray:
        return sps.block_diag([self.as_array(arr) for arr in arrs], format="csc")

    def identity(self, size: int) -> SPArray:
        return sps.eye_array(size, dtype=self.dtype, format="csc")

    def empty(self, shape: tuple[int, ...]) -> SPArray:
        return sps.csc_array(shape, dtype=self.dtype)

    def dataframe_to_array(self, df: DataFrame) -> SPArray:
        return sps.csc_array(df.astype(pd.SparseDtype(self.dtype, 0.0)).sparse.to_coo())

    def scale_rows(self, mat: SPArray, vec: NDArray) -> SPArray:
        mat = self._validate_matrix(self.as_array(mat))
        vec = self._validate_vector(self.as_dense_array(vec))
        return type(mat)(sps.diags_array(vec).dot(mat))

    def scale_cols(self, mat: SPArray, vec: NDArray) -> SPArray:
        mat = self._validate_matrix(self.as_array(mat))
        vec = self._validate_vector(self.as_dense_array(vec))
        return type(mat)(mat.dot(sps.diags_array(vec)))

    def solve(self, mat: SPArray, vec: NDArray, method="", **kwargs) -> NDArray:
        mat = self._validate_matrix(self.as_array(mat), square=True)
        vec = self._validate_vector(self.as_dense_array(vec))
        if method == "":
            soln = sps.linalg.spsolve(mat, vec, **kwargs)
        elif method == "cg":
            soln, info = sps.linalg.cg(mat, vec, **kwargs)
            if info > 0:
                raise RuntimeError(f"CG convergence not achieved. with {info=:}")
        else:
            raise ValueError(f"{method=:} is not supported.")
        return soln
