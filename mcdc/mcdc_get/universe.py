from numba import njit


@njit
def cell_index_all(universe, data):
    start = universe["cell_index_offset"]
    end = start + universe["N_cell_index"]
    return data[start:end]


@njit
def cell_index(index, universe, data):
    offset = universe["cell_index_offset"]
    return data[offset + index]


@njit
def cell_index_chunk(start, size, universe, data):
    start += universe["cell_index_offset"]
    end = start + size
    return data[start:end]
