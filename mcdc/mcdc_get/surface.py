from numba import njit


@njit
def move_time_grid(index, surface, data):
    offset = surface["move_time_grid_offset"]
    return data[offset + index]


@njit
def move_time_grid_last(surface, data):
    start = surface["move_time_grid_offset"]
    end = start + surface["move_time_grid_length"]
    return data[end - 1]


@njit
def move_time_grid_all(surface, data):
    start = surface["move_time_grid_offset"]
    end = start + surface["move_time_grid_length"]
    return data[start:end]


@njit
def move_time_grid_chunk(start, length, surface, data):
    start += surface["move_time_grid_offset"]
    end = start + length
    return data[start:end]


@njit
def move_translations_vector(index_1, surface, data):
    offset = surface["move_translations_offset"]
    stride = 3
    start = offset + index_1 * stride
    end = start + stride
    return data[start:end]


@njit
def move_translations(index_1, index_2, surface, data):
    offset = surface["move_translations_offset"]
    stride = 3
    return data[offset + index_1 * stride + index_2]


@njit
def move_translations_chunk(start, length, surface, data):
    start += surface["move_translations_offset"]
    end = start + length
    return data[start:end]


@njit
def move_velocities_vector(index_1, surface, data):
    offset = surface["move_velocities_offset"]
    stride = 3
    start = offset + index_1 * stride
    end = start + stride
    return data[start:end]


@njit
def move_velocities(index_1, index_2, surface, data):
    offset = surface["move_velocities_offset"]
    stride = 3
    return data[offset + index_1 * stride + index_2]


@njit
def move_velocities_chunk(start, length, surface, data):
    start += surface["move_velocities_offset"]
    end = start + length
    return data[start:end]


@njit
def tally_IDs(index, surface, data):
    offset = surface["tally_IDs_offset"]
    return data[offset + index]


@njit
def tally_IDs_last(surface, data):
    start = surface["tally_IDs_offset"]
    end = start + surface["tally_IDs_length"]
    return data[end - 1]


@njit
def tally_IDs_all(surface, data):
    start = surface["tally_IDs_offset"]
    end = start + surface["tally_IDs_length"]
    return data[start:end]


@njit
def tally_IDs_chunk(start, length, surface, data):
    start += surface["tally_IDs_offset"]
    end = start + length
    return data[start:end]
