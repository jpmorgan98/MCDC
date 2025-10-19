from numba import njit


@njit
def scores(index, global_tally, data, value):
    offset = global_tally["scores_offset"]
    data[offset + index] = value


@njit
def scores_all(global_tally, data, value):
    start = global_tally["scores_offset"]
    size = global_tally["scores_length"]
    end = start + size
    data[start:end] = value


@njit
def scores_last(global_tally, data, value):
    start = global_tally["scores_offset"]
    size = global_tally["scores_length"]
    end = start + size
    data[end - 1] = value


@njit
def scores_chunk(start, length, global_tally, data, value):
    start += global_tally["scores_offset"]
    end = start + length
    data[start:end] = value


@njit
def mu(index, global_tally, data, value):
    offset = global_tally["mu_offset"]
    data[offset + index] = value


@njit
def mu_all(global_tally, data, value):
    start = global_tally["mu_offset"]
    size = global_tally["mu_length"]
    end = start + size
    data[start:end] = value


@njit
def mu_last(global_tally, data, value):
    start = global_tally["mu_offset"]
    size = global_tally["mu_length"]
    end = start + size
    data[end - 1] = value


@njit
def mu_chunk(start, length, global_tally, data, value):
    start += global_tally["mu_offset"]
    end = start + length
    data[start:end] = value


@njit
def azi(index, global_tally, data, value):
    offset = global_tally["azi_offset"]
    data[offset + index] = value


@njit
def azi_all(global_tally, data, value):
    start = global_tally["azi_offset"]
    size = global_tally["azi_length"]
    end = start + size
    data[start:end] = value


@njit
def azi_last(global_tally, data, value):
    start = global_tally["azi_offset"]
    size = global_tally["azi_length"]
    end = start + size
    data[end - 1] = value


@njit
def azi_chunk(start, length, global_tally, data, value):
    start += global_tally["azi_offset"]
    end = start + length
    data[start:end] = value


@njit
def energy(index, global_tally, data, value):
    offset = global_tally["energy_offset"]
    data[offset + index] = value


@njit
def energy_all(global_tally, data, value):
    start = global_tally["energy_offset"]
    size = global_tally["energy_length"]
    end = start + size
    data[start:end] = value


@njit
def energy_last(global_tally, data, value):
    start = global_tally["energy_offset"]
    size = global_tally["energy_length"]
    end = start + size
    data[end - 1] = value


@njit
def energy_chunk(start, length, global_tally, data, value):
    start += global_tally["energy_offset"]
    end = start + length
    data[start:end] = value


@njit
def time(index, global_tally, data, value):
    offset = global_tally["time_offset"]
    data[offset + index] = value


@njit
def time_all(global_tally, data, value):
    start = global_tally["time_offset"]
    size = global_tally["time_length"]
    end = start + size
    data[start:end] = value


@njit
def time_last(global_tally, data, value):
    start = global_tally["time_offset"]
    size = global_tally["time_length"]
    end = start + size
    data[end - 1] = value


@njit
def time_chunk(start, length, global_tally, data, value):
    start += global_tally["time_offset"]
    end = start + length
    data[start:end] = value


@njit
def bin(index, global_tally, data, value):
    offset = global_tally["bin_offset"]
    data[offset + index] = value


@njit
def bin_all(global_tally, data, value):
    start = global_tally["bin_offset"]
    size = global_tally["bin_length"]
    end = start + size
    data[start:end] = value


@njit
def bin_last(global_tally, data, value):
    start = global_tally["bin_offset"]
    size = global_tally["bin_length"]
    end = start + size
    data[end - 1] = value


@njit
def bin_chunk(start, length, global_tally, data, value):
    start += global_tally["bin_offset"]
    end = start + length
    data[start:end] = value


@njit
def bin_sum(index, global_tally, data, value):
    offset = global_tally["bin_sum_offset"]
    data[offset + index] = value


@njit
def bin_sum_all(global_tally, data, value):
    start = global_tally["bin_sum_offset"]
    size = global_tally["bin_sum_length"]
    end = start + size
    data[start:end] = value


@njit
def bin_sum_last(global_tally, data, value):
    start = global_tally["bin_sum_offset"]
    size = global_tally["bin_sum_length"]
    end = start + size
    data[end - 1] = value


@njit
def bin_sum_chunk(start, length, global_tally, data, value):
    start += global_tally["bin_sum_offset"]
    end = start + length
    data[start:end] = value


@njit
def bin_sum_square(index, global_tally, data, value):
    offset = global_tally["bin_sum_square_offset"]
    data[offset + index] = value


@njit
def bin_sum_square_all(global_tally, data, value):
    start = global_tally["bin_sum_square_offset"]
    size = global_tally["bin_sum_square_length"]
    end = start + size
    data[start:end] = value


@njit
def bin_sum_square_last(global_tally, data, value):
    start = global_tally["bin_sum_square_offset"]
    size = global_tally["bin_sum_square_length"]
    end = start + size
    data[end - 1] = value


@njit
def bin_sum_square_chunk(start, length, global_tally, data, value):
    start += global_tally["bin_sum_square_offset"]
    end = start + length
    data[start:end] = value


@njit
def bin_shape(index, global_tally, data, value):
    offset = global_tally["bin_shape_offset"]
    data[offset + index] = value


@njit
def bin_shape_all(global_tally, data, value):
    start = global_tally["bin_shape_offset"]
    size = global_tally["bin_shape_length"]
    end = start + size
    data[start:end] = value


@njit
def bin_shape_last(global_tally, data, value):
    start = global_tally["bin_shape_offset"]
    size = global_tally["bin_shape_length"]
    end = start + size
    data[end - 1] = value


@njit
def bin_shape_chunk(start, length, global_tally, data, value):
    start += global_tally["bin_shape_offset"]
    end = start + length
    data[start:end] = value
