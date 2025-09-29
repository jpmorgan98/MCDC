from numba import njit


@njit
def xs_energy_grid(index, nuclide, data):
    offset = nuclide["xs_energy_grid_offset"]
    return data[offset + index]


@njit
def xs_energy_grid_last(nuclide, data):
    start = nuclide["xs_energy_grid_offset"]
    end = start + nuclide["xs_energy_grid_length"]
    return data[end - 1]


@njit
def xs_energy_grid_all(nuclide, data):
    start = nuclide["xs_energy_grid_offset"]
    end = start + nuclide["xs_energy_grid_length"]
    return data[start:end]


@njit
def xs_energy_grid_chunk(start, length, nuclide, data):
    start += nuclide["xs_energy_grid_offset"]
    end = start + length
    return data[start:end]


@njit
def total_xs(index, nuclide, data):
    offset = nuclide["total_xs_offset"]
    return data[offset + index]


@njit
def total_xs_last(nuclide, data):
    start = nuclide["total_xs_offset"]
    end = start + nuclide["total_xs_length"]
    return data[end - 1]


@njit
def total_xs_all(nuclide, data):
    start = nuclide["total_xs_offset"]
    end = start + nuclide["total_xs_length"]
    return data[start:end]


@njit
def total_xs_chunk(start, length, nuclide, data):
    start += nuclide["total_xs_offset"]
    end = start + length
    return data[start:end]


@njit
def reaction_types(index, nuclide, data):
    offset = nuclide["reaction_types_offset"]
    return data[offset + index]


@njit
def reaction_types_last(nuclide, data):
    start = nuclide["reaction_types_offset"]
    end = start + nuclide["reaction_types_length"]
    return data[end - 1]


@njit
def reaction_types_all(nuclide, data):
    start = nuclide["reaction_types_offset"]
    end = start + nuclide["reaction_types_length"]
    return data[start:end]


@njit
def reaction_types_chunk(start, length, nuclide, data):
    start += nuclide["reaction_types_offset"]
    end = start + length
    return data[start:end]


@njit
def reaction_IDs(index, nuclide, data):
    offset = nuclide["reaction_IDs_offset"]
    return data[offset + index]


@njit
def reaction_IDs_last(nuclide, data):
    start = nuclide["reaction_IDs_offset"]
    end = start + nuclide["reaction_IDs_length"]
    return data[end - 1]


@njit
def reaction_IDs_all(nuclide, data):
    start = nuclide["reaction_IDs_offset"]
    end = start + nuclide["reaction_IDs_length"]
    return data[start:end]


@njit
def reaction_IDs_chunk(start, length, nuclide, data):
    start += nuclide["reaction_IDs_offset"]
    end = start + length
    return data[start:end]
