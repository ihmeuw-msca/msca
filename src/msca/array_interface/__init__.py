from typing import Literal

from msca.array_interface.jax import JAXArrayInterface
from msca.array_interface.main import ArrayInterface
from msca.array_interface.numpy import NumpyArrayInterface
from msca.array_interface.sparse import SparseArrayInterface


def build_array_interface(
    atype: Literal["jax", "numpy", "sparse"], dtype=Literal["float32", "float64"]
) -> ArrayInterface:
    if atype not in ["jax", "numpy", "sparse"]:
        raise ValueError(f"Invalid array type: {atype}")
    if atype == "jax":
        return JAXArrayInterface(dtype)
    if atype == "numpy":
        return NumpyArrayInterface(dtype)
    if atype == "sparse":
        return SparseArrayInterface(dtype)
