from numba import njit


@njit
def scores(index, surface_tally, data):
    offset = surface_tally["scores_offset"]
    return data[offset + index]


@njit
def scores_all(surface_tally, data):
    start = surface_tally["scores_offset"]
    size = surface_tally["scores_length"]
    end = start + size
    return data[start:end]


@njit
def scores_last(surface_tally, data):
    start = surface_tally["scores_offset"]
    size = surface_tally["scores_length"]
    end = start + size
    return data[end - 1]


@njit
def scores_chunk(start, length, surface_tally, data):
    start += surface_tally["scores_offset"]
    end = start + length
    return data[start:end]


@njit
def mu(index, surface_tally, data):
    offset = surface_tally["mu_offset"]
    return data[offset + index]


@njit
def mu_all(surface_tally, data):
    start = surface_tally["mu_offset"]
    size = surface_tally["mu_length"]
    end = start + size
    return data[start:end]


@njit
def mu_last(surface_tally, data):
    start = surface_tally["mu_offset"]
    size = surface_tally["mu_length"]
    end = start + size
    return data[end - 1]


@njit
def mu_chunk(start, length, surface_tally, data):
    start += surface_tally["mu_offset"]
    end = start + length
    return data[start:end]


@njit
def azi(index, surface_tally, data):
    offset = surface_tally["azi_offset"]
    return data[offset + index]


@njit
def azi_all(surface_tally, data):
    start = surface_tally["azi_offset"]
    size = surface_tally["azi_length"]
    end = start + size
    return data[start:end]


@njit
def azi_last(surface_tally, data):
    start = surface_tally["azi_offset"]
    size = surface_tally["azi_length"]
    end = start + size
    return data[end - 1]


@njit
def azi_chunk(start, length, surface_tally, data):
    start += surface_tally["azi_offset"]
    end = start + length
    return data[start:end]


@njit
def energy(index, surface_tally, data):
    offset = surface_tally["energy_offset"]
    return data[offset + index]


@njit
def energy_all(surface_tally, data):
    start = surface_tally["energy_offset"]
    size = surface_tally["energy_length"]
    end = start + size
    return data[start:end]


@njit
def energy_last(surface_tally, data):
    start = surface_tally["energy_offset"]
    size = surface_tally["energy_length"]
    end = start + size
    return data[end - 1]


@njit
def energy_chunk(start, length, surface_tally, data):
    start += surface_tally["energy_offset"]
    end = start + length
    return data[start:end]


@njit
def time(index, surface_tally, data):
    offset = surface_tally["time_offset"]
    return data[offset + index]


@njit
def time_all(surface_tally, data):
    start = surface_tally["time_offset"]
    size = surface_tally["time_length"]
    end = start + size
    return data[start:end]


@njit
def time_last(surface_tally, data):
    start = surface_tally["time_offset"]
    size = surface_tally["time_length"]
    end = start + size
    return data[end - 1]


@njit
def time_chunk(start, length, surface_tally, data):
    start += surface_tally["time_offset"]
    end = start + length
    return data[start:end]


@njit
def bin(index, surface_tally, data):
    offset = surface_tally["bin_offset"]
    return data[offset + index]


@njit
def bin_all(surface_tally, data):
    start = surface_tally["bin_offset"]
    size = surface_tally["bin_length"]
    end = start + size
    return data[start:end]


@njit
def bin_last(surface_tally, data):
    start = surface_tally["bin_offset"]
    size = surface_tally["bin_length"]
    end = start + size
    return data[end - 1]


@njit
def bin_chunk(start, length, surface_tally, data):
    start += surface_tally["bin_offset"]
    end = start + length
    return data[start:end]


@njit
def bin_sum(index, surface_tally, data):
    offset = surface_tally["bin_sum_offset"]
    return data[offset + index]


@njit
def bin_sum_all(surface_tally, data):
    start = surface_tally["bin_sum_offset"]
    size = surface_tally["bin_sum_length"]
    end = start + size
    return data[start:end]


@njit
def bin_sum_last(surface_tally, data):
    start = surface_tally["bin_sum_offset"]
    size = surface_tally["bin_sum_length"]
    end = start + size
    return data[end - 1]


@njit
def bin_sum_chunk(start, length, surface_tally, data):
    start += surface_tally["bin_sum_offset"]
    end = start + length
    return data[start:end]


@njit
def bin_sum_square(index, surface_tally, data):
    offset = surface_tally["bin_sum_square_offset"]
    return data[offset + index]


@njit
def bin_sum_square_all(surface_tally, data):
    start = surface_tally["bin_sum_square_offset"]
    size = surface_tally["bin_sum_square_length"]
    end = start + size
    return data[start:end]


@njit
def bin_sum_square_last(surface_tally, data):
    start = surface_tally["bin_sum_square_offset"]
    size = surface_tally["bin_sum_square_length"]
    end = start + size
    return data[end - 1]


@njit
def bin_sum_square_chunk(start, length, surface_tally, data):
    start += surface_tally["bin_sum_square_offset"]
    end = start + length
    return data[start:end]


@njit
def bin_shape(index, surface_tally, data):
    offset = surface_tally["bin_shape_offset"]
    return data[offset + index]


@njit
def bin_shape_all(surface_tally, data):
    start = surface_tally["bin_shape_offset"]
    size = surface_tally["bin_shape_length"]
    end = start + size
    return data[start:end]


@njit
def bin_shape_last(surface_tally, data):
    start = surface_tally["bin_shape_offset"]
    size = surface_tally["bin_shape_length"]
    end = start + size
    return data[end - 1]


@njit
def bin_shape_chunk(start, length, surface_tally, data):
    start += surface_tally["bin_shape_offset"]
    end = start + length
    return data[start:end]
