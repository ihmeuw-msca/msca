import numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def build_indices_midpoint(long[::1] lb_index, long[::1] ub_index, long size):
    cdef int nrow = lb_index.size
    cdef int i = 0
    cdef int j = 0
    cdef int k = 0

    row_index = np.empty(size, dtype=int)
    col_index = np.empty(size, dtype=int)

    cdef long[::1] row_index_view = row_index
    cdef long[::1] col_index_view = col_index

    for i in range(nrow):
        for j in range(lb_index[i], ub_index[i]):
            row_index_view[k] = i
            col_index_view[k] = j
            k += 1

    return row_index, col_index
