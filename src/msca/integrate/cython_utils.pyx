import numpy as np
cimport cython
from libc.stdint cimport int64_t


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_midpoint(int64_t[::1] lb_index, int64_t[::1] ub_index, int64_t size):
    cdef int64_t nrow = lb_index.size
    cdef int64_t i = 0
    cdef int64_t j = 0
    cdef int64_t k = 0

    row_index = np.empty(size, dtype=np.int64)
    col_index = np.empty(size, dtype=np.int64)

    cdef int64_t[::1] row_index_view = row_index
    cdef int64_t[::1] col_index_view = col_index

    for i in range(nrow):
        for j in range(lb_index[i], ub_index[i]):
            row_index_view[k] = i
            col_index_view[k] = j
            k += 1

    return row_index, col_index
