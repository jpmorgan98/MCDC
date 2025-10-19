from numba import njit


@njit
def scores(index, mesh_tally, data, value):
    offset = mesh_tally["scores_offset"]
    data[offset + index] = value


@njit
def scores_all(mesh_tally, data, value):
    start = mesh_tally["scores_offset"]
    size = mesh_tally["scores_length"]
    end = start + size
    data[start:end] = value


@njit
def scores_last(mesh_tally, data, value):
    start = mesh_tally["scores_offset"]
    size = mesh_tally["scores_length"]
    end = start + size
    data[end - 1] = value


@njit
def scores_chunk(start, length, mesh_tally, data, value):
    start += mesh_tally["scores_offset"]
    end = start + length
    data[start:end] = value


@njit
def mu(index, mesh_tally, data, value):
    offset = mesh_tally["mu_offset"]
    data[offset + index] = value


@njit
def mu_all(mesh_tally, data, value):
    start = mesh_tally["mu_offset"]
    size = mesh_tally["mu_length"]
    end = start + size
    data[start:end] = value


@njit
def mu_last(mesh_tally, data, value):
    start = mesh_tally["mu_offset"]
    size = mesh_tally["mu_length"]
    end = start + size
    data[end - 1] = value


@njit
def mu_chunk(start, length, mesh_tally, data, value):
    start += mesh_tally["mu_offset"]
    end = start + length
    data[start:end] = value


@njit
def azi(index, mesh_tally, data, value):
    offset = mesh_tally["azi_offset"]
    data[offset + index] = value


@njit
def azi_all(mesh_tally, data, value):
    start = mesh_tally["azi_offset"]
    size = mesh_tally["azi_length"]
    end = start + size
    data[start:end] = value


@njit
def azi_last(mesh_tally, data, value):
    start = mesh_tally["azi_offset"]
    size = mesh_tally["azi_length"]
    end = start + size
    data[end - 1] = value


@njit
def azi_chunk(start, length, mesh_tally, data, value):
    start += mesh_tally["azi_offset"]
    end = start + length
    data[start:end] = value


@njit
def energy(index, mesh_tally, data, value):
    offset = mesh_tally["energy_offset"]
    data[offset + index] = value


@njit
def energy_all(mesh_tally, data, value):
    start = mesh_tally["energy_offset"]
    size = mesh_tally["energy_length"]
    end = start + size
    data[start:end] = value


@njit
def energy_last(mesh_tally, data, value):
    start = mesh_tally["energy_offset"]
    size = mesh_tally["energy_length"]
    end = start + size
    data[end - 1] = value


@njit
def energy_chunk(start, length, mesh_tally, data, value):
    start += mesh_tally["energy_offset"]
    end = start + length
    data[start:end] = value


@njit
def time(index, mesh_tally, data, value):
    offset = mesh_tally["time_offset"]
    data[offset + index] = value


@njit
def time_all(mesh_tally, data, value):
    start = mesh_tally["time_offset"]
    size = mesh_tally["time_length"]
    end = start + size
    data[start:end] = value


@njit
def time_last(mesh_tally, data, value):
    start = mesh_tally["time_offset"]
    size = mesh_tally["time_length"]
    end = start + size
    data[end - 1] = value


@njit
def time_chunk(start, length, mesh_tally, data, value):
    start += mesh_tally["time_offset"]
    end = start + length
    data[start:end] = value


@njit
def bin(index, mesh_tally, data, value):
    offset = mesh_tally["bin_offset"]
    data[offset + index] = value


@njit
def bin_all(mesh_tally, data, value):
    start = mesh_tally["bin_offset"]
    size = mesh_tally["bin_length"]
    end = start + size
    data[start:end] = value


@njit
def bin_last(mesh_tally, data, value):
    start = mesh_tally["bin_offset"]
    size = mesh_tally["bin_length"]
    end = start + size
    data[end - 1] = value


@njit
def bin_chunk(start, length, mesh_tally, data, value):
    start += mesh_tally["bin_offset"]
    end = start + length
    data[start:end] = value


@njit
def bin_sum(index, mesh_tally, data, value):
    offset = mesh_tally["bin_sum_offset"]
    data[offset + index] = value


@njit
def bin_sum_all(mesh_tally, data, value):
    start = mesh_tally["bin_sum_offset"]
    size = mesh_tally["bin_sum_length"]
    end = start + size
    data[start:end] = value


@njit
def bin_sum_last(mesh_tally, data, value):
    start = mesh_tally["bin_sum_offset"]
    size = mesh_tally["bin_sum_length"]
    end = start + size
    data[end - 1] = value


@njit
def bin_sum_chunk(start, length, mesh_tally, data, value):
    start += mesh_tally["bin_sum_offset"]
    end = start + length
    data[start:end] = value


@njit
def bin_sum_square(index, mesh_tally, data, value):
    offset = mesh_tally["bin_sum_square_offset"]
    data[offset + index] = value


@njit
def bin_sum_square_all(mesh_tally, data, value):
    start = mesh_tally["bin_sum_square_offset"]
    size = mesh_tally["bin_sum_square_length"]
    end = start + size
    data[start:end] = value


@njit
def bin_sum_square_last(mesh_tally, data, value):
    start = mesh_tally["bin_sum_square_offset"]
    size = mesh_tally["bin_sum_square_length"]
    end = start + size
    data[end - 1] = value


@njit
def bin_sum_square_chunk(start, length, mesh_tally, data, value):
    start += mesh_tally["bin_sum_square_offset"]
    end = start + length
    data[start:end] = value


@njit
def bin_shape(index, mesh_tally, data, value):
    offset = mesh_tally["bin_shape_offset"]
    data[offset + index] = value


@njit
def bin_shape_all(mesh_tally, data, value):
    start = mesh_tally["bin_shape_offset"]
    size = mesh_tally["bin_shape_length"]
    end = start + size
    data[start:end] = value


@njit
def bin_shape_last(mesh_tally, data, value):
    start = mesh_tally["bin_shape_offset"]
    size = mesh_tally["bin_shape_length"]
    end = start + size
    data[end - 1] = value


@njit
def bin_shape_chunk(start, length, mesh_tally, data, value):
    start += mesh_tally["bin_shape_offset"]
    end = start + length
    data[start:end] = value
