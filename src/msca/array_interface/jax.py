import jax.numpy as jnp
import jax.scipy as jsp
from pandas import DataFrame

from msca.array_interface.main import ArrayInterface
from msca.typing import Array, JAXArray, SPArray


class JAXArrayInterface(ArrayInterface):
    def as_array(self, arr: Array) -> JAXArray:
        arr = self._validate_array(arr)
        if isinstance(arr, SPArray):
            arr = arr.toarray()
        return jnp.asarray(arr, dtype=self.dtype)

    def as_dense_array(self, arr: Array) -> JAXArray:
        return self.as_array(arr)

    def vstack(self, arrs: list[Array]) -> JAXArray:
        return jnp.vstack([self.as_array(arr) for arr in arrs])

    def vstack_dense_array(self, arrs: list[Array]) -> JAXArray:
        return self.vstack(arrs)

    def hstack(self, arrs: list[Array]) -> JAXArray:
        return jnp.hstack([self.as_array(arr) for arr in arrs])

    def hstack_dense_array(self, arrs: list[Array]) -> JAXArray:
        return self.hstack(arrs)

    def block_diag(self, arrs: list[Array]) -> JAXArray:
        return jsp.linalg.block_diag(*[self.as_array(arr) for arr in arrs])

    def identity(self, size: int) -> JAXArray:
        return jnp.identity(size, dtype=self.dtype)

    def empty(self, shape: tuple[int, ...]) -> JAXArray:
        return jnp.empty(shape, dtype=self.dtype)

    def dataframe_to_array(self, df: DataFrame) -> JAXArray:
        return self.as_array(df.to_numpy())

    def scale_rows(self, mat: JAXArray, vec: JAXArray) -> JAXArray:
        mat = self._validate_matrix(self.as_array(mat))
        vec = self._validate_vector(self.as_dense_array(vec))
        return vec[:, jnp.newaxis] * mat

    def scale_cols(self, mat: JAXArray, vec: JAXArray) -> JAXArray:
        mat = self._validate_matrix(self.as_array(mat))
        vec = self._validate_vector(self.as_dense_array(vec))
        return mat * vec

    def solve(self, mat: JAXArray, vec: JAXArray, method="", **kwargs) -> JAXArray:
        mat = self._validate_matrix(self.as_array(mat), square=True)
        vec = self._validate_vector(self.as_dense_array(vec))
        if method == "":
            soln = jnp.linalg.solve(mat, vec, **kwargs)
        elif method == "cg":
            soln, info = jsp.sparse.linalg.cg(mat, vec, **kwargs)
            if info > 0:
                raise RuntimeError(f"CG convergence not achieved. with {info=:}")
        else:
            raise ValueError(f"{method=:} is not supported.")
        return soln
