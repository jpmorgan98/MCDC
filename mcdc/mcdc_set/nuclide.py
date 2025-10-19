from numba import njit


@njit
def xs_energy_grid(index, nuclide, data, value):
    offset = nuclide["xs_energy_grid_offset"]
    data[offset + index] = value


@njit
def xs_energy_grid_all(nuclide, data, value):
    start = nuclide["xs_energy_grid_offset"]
    size = nuclide["xs_energy_grid_length"]
    end = start + size
    data[start:end] = value


@njit
def xs_energy_grid_last(nuclide, data, value):
    start = nuclide["xs_energy_grid_offset"]
    size = nuclide["xs_energy_grid_length"]
    end = start + size
    data[end - 1] = value


@njit
def xs_energy_grid_chunk(start, length, nuclide, data, value):
    start += nuclide["xs_energy_grid_offset"]
    end = start + length
    data[start:end] = value


@njit
def reaction_IDs(index, nuclide, data, value):
    offset = nuclide["reaction_IDs_offset"]
    data[offset + index] = value


@njit
def reaction_IDs_all(nuclide, data, value):
    start = nuclide["reaction_IDs_offset"]
    size = nuclide["N_reaction"]
    end = start + size
    data[start:end] = value


@njit
def reaction_IDs_last(nuclide, data, value):
    start = nuclide["reaction_IDs_offset"]
    size = nuclide["N_reaction"]
    end = start + size
    data[end - 1] = value


@njit
def reaction_IDs_chunk(start, length, nuclide, data, value):
    start += nuclide["reaction_IDs_offset"]
    end = start + length
    data[start:end] = value


@njit
def reaction_types(index, nuclide, data, value):
    offset = nuclide["reaction_types_offset"]
    data[offset + index] = value


@njit
def reaction_types_all(nuclide, data, value):
    start = nuclide["reaction_types_offset"]
    size = nuclide["N_reaction"]
    end = start + size
    data[start:end] = value


@njit
def reaction_types_last(nuclide, data, value):
    start = nuclide["reaction_types_offset"]
    size = nuclide["N_reaction"]
    end = start + size
    data[end - 1] = value


@njit
def reaction_types_chunk(start, length, nuclide, data, value):
    start += nuclide["reaction_types_offset"]
    end = start + length
    data[start:end] = value


@njit
def total_xs(index, nuclide, data, value):
    offset = nuclide["total_xs_offset"]
    data[offset + index] = value


@njit
def total_xs_all(nuclide, data, value):
    start = nuclide["total_xs_offset"]
    size = nuclide["total_xs_length"]
    end = start + size
    data[start:end] = value


@njit
def total_xs_last(nuclide, data, value):
    start = nuclide["total_xs_offset"]
    size = nuclide["total_xs_length"]
    end = start + size
    data[end - 1] = value


@njit
def total_xs_chunk(start, length, nuclide, data, value):
    start += nuclide["total_xs_offset"]
    end = start + length
    data[start:end] = value
