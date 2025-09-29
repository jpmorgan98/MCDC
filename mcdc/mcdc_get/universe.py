from numba import njit


@njit
def cell_IDs(index, universe, data):
    offset = universe["cell_IDs_offset"]
    return data[offset + index]


@njit
def cell_IDs_last(universe, data):
    start = universe["cell_IDs_offset"]
    end = start + universe["cell_IDs_length"]
    return data[end - 1]


@njit
def cell_IDs_all(universe, data):
    start = universe["cell_IDs_offset"]
    end = start + universe["cell_IDs_length"]
    return data[start:end]


@njit
def cell_IDs_chunk(start, length, universe, data):
    start += universe["cell_IDs_offset"]
    end = start + length
    return data[start:end]
