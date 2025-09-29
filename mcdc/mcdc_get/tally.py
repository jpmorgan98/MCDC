from numba import njit


@njit
def mu(index, tally, data):
    offset = tally["mu_offset"]
    return data[offset + index]


@njit
def mu_last(tally, data):
    start = tally["mu_offset"]
    end = start + tally["mu_length"]
    return data[end - 1]


@njit
def mu_all(tally, data):
    start = tally["mu_offset"]
    end = start + tally["mu_length"]
    return data[start:end]


@njit
def mu_chunk(start, length, tally, data):
    start += tally["mu_offset"]
    end = start + length
    return data[start:end]


@njit
def azi(index, tally, data):
    offset = tally["azi_offset"]
    return data[offset + index]


@njit
def azi_last(tally, data):
    start = tally["azi_offset"]
    end = start + tally["azi_length"]
    return data[end - 1]


@njit
def azi_all(tally, data):
    start = tally["azi_offset"]
    end = start + tally["azi_length"]
    return data[start:end]


@njit
def azi_chunk(start, length, tally, data):
    start += tally["azi_offset"]
    end = start + length
    return data[start:end]


@njit
def polar_reference(index, tally, data):
    offset = tally["polar_reference_offset"]
    return data[offset + index]


@njit
def polar_reference_last(tally, data):
    start = tally["polar_reference_offset"]
    end = start + tally["polar_reference_length"]
    return data[end - 1]


@njit
def polar_reference_all(tally, data):
    start = tally["polar_reference_offset"]
    end = start + tally["polar_reference_length"]
    return data[start:end]


@njit
def polar_reference_chunk(start, length, tally, data):
    start += tally["polar_reference_offset"]
    end = start + length
    return data[start:end]


@njit
def energy(index, tally, data):
    offset = tally["energy_offset"]
    return data[offset + index]


@njit
def energy_last(tally, data):
    start = tally["energy_offset"]
    end = start + tally["energy_length"]
    return data[end - 1]


@njit
def energy_all(tally, data):
    start = tally["energy_offset"]
    end = start + tally["energy_length"]
    return data[start:end]


@njit
def energy_chunk(start, length, tally, data):
    start += tally["energy_offset"]
    end = start + length
    return data[start:end]


@njit
def time(index, tally, data):
    offset = tally["time_offset"]
    return data[offset + index]


@njit
def time_last(tally, data):
    start = tally["time_offset"]
    end = start + tally["time_length"]
    return data[end - 1]


@njit
def time_all(tally, data):
    start = tally["time_offset"]
    end = start + tally["time_length"]
    return data[start:end]


@njit
def time_chunk(start, length, tally, data):
    start += tally["time_offset"]
    end = start + length
    return data[start:end]


@njit
def x(index, tally, data):
    offset = tally["x_offset"]
    return data[offset + index]


@njit
def x_last(tally, data):
    start = tally["x_offset"]
    end = start + tally["x_length"]
    return data[end - 1]


@njit
def x_all(tally, data):
    start = tally["x_offset"]
    end = start + tally["x_length"]
    return data[start:end]


@njit
def x_chunk(start, length, tally, data):
    start += tally["x_offset"]
    end = start + length
    return data[start:end]


@njit
def y(index, tally, data):
    offset = tally["y_offset"]
    return data[offset + index]


@njit
def y_last(tally, data):
    start = tally["y_offset"]
    end = start + tally["y_length"]
    return data[end - 1]


@njit
def y_all(tally, data):
    start = tally["y_offset"]
    end = start + tally["y_length"]
    return data[start:end]


@njit
def y_chunk(start, length, tally, data):
    start += tally["y_offset"]
    end = start + length
    return data[start:end]


@njit
def z(index, tally, data):
    offset = tally["z_offset"]
    return data[offset + index]


@njit
def z_last(tally, data):
    start = tally["z_offset"]
    end = start + tally["z_length"]
    return data[end - 1]


@njit
def z_all(tally, data):
    start = tally["z_offset"]
    end = start + tally["z_length"]
    return data[start:end]


@njit
def z_chunk(start, length, tally, data):
    start += tally["z_offset"]
    end = start + length
    return data[start:end]


@njit
def scores(index, tally, data):
    offset = tally["scores_offset"]
    return data[offset + index]


@njit
def scores_last(tally, data):
    start = tally["scores_offset"]
    end = start + tally["scores_length"]
    return data[end - 1]


@njit
def scores_all(tally, data):
    start = tally["scores_offset"]
    end = start + tally["scores_length"]
    return data[start:end]


@njit
def scores_chunk(start, length, tally, data):
    start += tally["scores_offset"]
    end = start + length
    return data[start:end]
