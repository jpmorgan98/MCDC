from numba import njit


@njit
def universe_IDs(index, lattice, data, value):
    offset = lattice["universe_IDs_offset"]
    data[offset + index] = value


@njit
def universe_IDs_all(lattice, data, value):
    start = lattice["universe_IDs_offset"]
    size = lattice["N_universe"]
    end = start + size
    data[start:end] = value


@njit
def universe_IDs_last(lattice, data, value):
    start = lattice["universe_IDs_offset"]
    size = lattice["N_universe"]
    end = start + size
    data[end - 1] = value


@njit
def universe_IDs_chunk(start, length, lattice, data, value):
    start += lattice["universe_IDs_offset"]
    end = start + length
    data[start:end] = value
