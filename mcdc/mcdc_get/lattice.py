from numba import njit


@njit
def universe_index(index_1, index_2, index_3, lattice, data):
    offset = lattice["universe_index_offset"]
    stride_2 = lattice["Ny"]
    stride_3 = lattice["Nz"]
    return data[offset + index_1 * stride_2 * stride_3 + index_2 * stride_3 + index_3]


@njit
def universe_index_chunk(start, size, lattice, data):
    start += lattice["universe_index_offset"]
    end = start + size
    return data[start:end]
