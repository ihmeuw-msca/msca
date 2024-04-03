import jax
import numpy as np
import scipy.sparse as sps

NDArray = np.ndarray
JAXArray = jax.Array
SPArray = sps.sparray

Array = NDArray | JAXArray | SPArray
DenseArray = NDArray | JAXArray
