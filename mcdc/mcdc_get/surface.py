from numba import njit


@njit
def from_cell(index, cell, mcdc, data):
    offset = cell["surface_index_offset"]
    surface_ID = int(data[offset + index])
    return mcdc["surfaces"][surface_ID]
