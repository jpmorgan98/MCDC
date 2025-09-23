from numba import njit


@njit
def xs_energy_grid_all(nuclide, data):
    start = nuclide["xs_energy_grid_offset"]
    end = start + nuclide["N_xs_energy_grid"]
    return data[start:end]


@njit
def xs_energy_grid(index, nuclide, data):
    offset = nuclide["xs_energy_grid_offset"]
    return data[offset + index]


@njit
def xs_energy_grid_chunk(start, size, nuclide, data):
    start += nuclide["xs_energy_grid_offset"]
    end = start + size
    return data[start:end]


@njit
def total_xs_all(nuclide, data):
    start = nuclide["total_xs_offset"]
    end = start + nuclide["N_total_xs"]
    return data[start:end]


@njit
def total_xs(index, nuclide, data):
    offset = nuclide["total_xs_offset"]
    return data[offset + index]


@njit
def total_xs_chunk(start, size, nuclide, data):
    start += nuclide["total_xs_offset"]
    end = start + size
    return data[start:end]


@njit
def reaction_type_all(nuclide, data):
    start = nuclide["reaction_type_offset"]
    end = start + nuclide["N_reaction_type"]
    return data[start:end]


@njit
def reaction_type(index, nuclide, data):
    offset = nuclide["reaction_type_offset"]
    return data[offset + index]


@njit
def reaction_type_chunk(start, size, nuclide, data):
    start += nuclide["reaction_type_offset"]
    end = start + size
    return data[start:end]


@njit
def reaction_index_all(nuclide, data):
    start = nuclide["reaction_index_offset"]
    end = start + nuclide["N_reaction_index"]
    return data[start:end]


@njit
def reaction_index(index, nuclide, data):
    offset = nuclide["reaction_index_offset"]
    return data[offset + index]


@njit
def reaction_index_chunk(start, size, nuclide, data):
    start += nuclide["reaction_index_offset"]
    end = start + size
    return data[start:end]
