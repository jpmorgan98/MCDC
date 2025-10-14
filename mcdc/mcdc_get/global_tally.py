from numba import njit


@njit
def scores(index, global_tally, data):
    offset = global_tally["scores_offset"]
    return data[offset + index]


@njit
def scores_all(global_tally, data):
    start = global_tally["scores_offset"]
    size = global_tally["scores_length"]
    end = start + size
    return data[start:end]


@njit
def scores_last(global_tally, data):
    start = global_tally["scores_offset"]
    size = global_tally["scores_length"]
    end = start + size
    return data[end - 1]


@njit
def scores_chunk(start, length, global_tally, data):
    start += global_tally["scores_offset"]
    end = start + length
    return data[start:end]


@njit
def mu(index, global_tally, data):
    offset = global_tally["mu_offset"]
    return data[offset + index]


@njit
def mu_all(global_tally, data):
    start = global_tally["mu_offset"]
    size = global_tally["mu_length"]
    end = start + size
    return data[start:end]


@njit
def mu_last(global_tally, data):
    start = global_tally["mu_offset"]
    size = global_tally["mu_length"]
    end = start + size
    return data[end - 1]


@njit
def mu_chunk(start, length, global_tally, data):
    start += global_tally["mu_offset"]
    end = start + length
    return data[start:end]


@njit
def azi(index, global_tally, data):
    offset = global_tally["azi_offset"]
    return data[offset + index]


@njit
def azi_all(global_tally, data):
    start = global_tally["azi_offset"]
    size = global_tally["azi_length"]
    end = start + size
    return data[start:end]


@njit
def azi_last(global_tally, data):
    start = global_tally["azi_offset"]
    size = global_tally["azi_length"]
    end = start + size
    return data[end - 1]


@njit
def azi_chunk(start, length, global_tally, data):
    start += global_tally["azi_offset"]
    end = start + length
    return data[start:end]


@njit
def energy(index, global_tally, data):
    offset = global_tally["energy_offset"]
    return data[offset + index]


@njit
def energy_all(global_tally, data):
    start = global_tally["energy_offset"]
    size = global_tally["energy_length"]
    end = start + size
    return data[start:end]


@njit
def energy_last(global_tally, data):
    start = global_tally["energy_offset"]
    size = global_tally["energy_length"]
    end = start + size
    return data[end - 1]


@njit
def energy_chunk(start, length, global_tally, data):
    start += global_tally["energy_offset"]
    end = start + length
    return data[start:end]


@njit
def time(index, global_tally, data):
    offset = global_tally["time_offset"]
    return data[offset + index]


@njit
def time_all(global_tally, data):
    start = global_tally["time_offset"]
    size = global_tally["time_length"]
    end = start + size
    return data[start:end]


@njit
def time_last(global_tally, data):
    start = global_tally["time_offset"]
    size = global_tally["time_length"]
    end = start + size
    return data[end - 1]


@njit
def time_chunk(start, length, global_tally, data):
    start += global_tally["time_offset"]
    end = start + length
    return data[start:end]


@njit
def bin(index, global_tally, data):
    offset = global_tally["bin_offset"]
    return data[offset + index]


@njit
def bin_all(global_tally, data):
    start = global_tally["bin_offset"]
    size = global_tally["bin_length"]
    end = start + size
    return data[start:end]


@njit
def bin_last(global_tally, data):
    start = global_tally["bin_offset"]
    size = global_tally["bin_length"]
    end = start + size
    return data[end - 1]


@njit
def bin_chunk(start, length, global_tally, data):
    start += global_tally["bin_offset"]
    end = start + length
    return data[start:end]


@njit
def bin_sum(index, global_tally, data):
    offset = global_tally["bin_sum_offset"]
    return data[offset + index]


@njit
def bin_sum_all(global_tally, data):
    start = global_tally["bin_sum_offset"]
    size = global_tally["bin_sum_length"]
    end = start + size
    return data[start:end]


@njit
def bin_sum_last(global_tally, data):
    start = global_tally["bin_sum_offset"]
    size = global_tally["bin_sum_length"]
    end = start + size
    return data[end - 1]


@njit
def bin_sum_chunk(start, length, global_tally, data):
    start += global_tally["bin_sum_offset"]
    end = start + length
    return data[start:end]


@njit
def bin_sum_square(index, global_tally, data):
    offset = global_tally["bin_sum_square_offset"]
    return data[offset + index]


@njit
def bin_sum_square_all(global_tally, data):
    start = global_tally["bin_sum_square_offset"]
    size = global_tally["bin_sum_square_length"]
    end = start + size
    return data[start:end]


@njit
def bin_sum_square_last(global_tally, data):
    start = global_tally["bin_sum_square_offset"]
    size = global_tally["bin_sum_square_length"]
    end = start + size
    return data[end - 1]


@njit
def bin_sum_square_chunk(start, length, global_tally, data):
    start += global_tally["bin_sum_square_offset"]
    end = start + length
    return data[start:end]
