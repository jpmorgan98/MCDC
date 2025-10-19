from numba import njit


@njit
def grid(index, multipdf_distribution, data, value):
    offset = multipdf_distribution["grid_offset"]
    data[offset + index] = value


@njit
def grid_all(multipdf_distribution, data, value):
    start = multipdf_distribution["grid_offset"]
    size = multipdf_distribution["grid_length"]
    end = start + size
    data[start:end] = value


@njit
def grid_last(multipdf_distribution, data, value):
    start = multipdf_distribution["grid_offset"]
    size = multipdf_distribution["grid_length"]
    end = start + size
    data[end - 1] = value


@njit
def grid_chunk(start, length, multipdf_distribution, data, value):
    start += multipdf_distribution["grid_offset"]
    end = start + length
    data[start:end] = value


@njit
def offset(index, multipdf_distribution, data, value):
    offset = multipdf_distribution["offset_offset"]
    data[offset + index] = value


@njit
def offset_all(multipdf_distribution, data, value):
    start = multipdf_distribution["offset_offset"]
    size = multipdf_distribution["offset_length"]
    end = start + size
    data[start:end] = value


@njit
def offset_last(multipdf_distribution, data, value):
    start = multipdf_distribution["offset_offset"]
    size = multipdf_distribution["offset_length"]
    end = start + size
    data[end - 1] = value


@njit
def offset_chunk(start, length, multipdf_distribution, data, value):
    start += multipdf_distribution["offset_offset"]
    end = start + length
    data[start:end] = value


@njit
def value(index, multipdf_distribution, data, value):
    offset = multipdf_distribution["value_offset"]
    data[offset + index] = value


@njit
def value_all(multipdf_distribution, data, value):
    start = multipdf_distribution["value_offset"]
    size = multipdf_distribution["value_length"]
    end = start + size
    data[start:end] = value


@njit
def value_last(multipdf_distribution, data, value):
    start = multipdf_distribution["value_offset"]
    size = multipdf_distribution["value_length"]
    end = start + size
    data[end - 1] = value


@njit
def value_chunk(start, length, multipdf_distribution, data, value):
    start += multipdf_distribution["value_offset"]
    end = start + length
    data[start:end] = value


@njit
def pdf(index, multipdf_distribution, data, value):
    offset = multipdf_distribution["pdf_offset"]
    data[offset + index] = value


@njit
def pdf_all(multipdf_distribution, data, value):
    start = multipdf_distribution["pdf_offset"]
    size = multipdf_distribution["pdf_length"]
    end = start + size
    data[start:end] = value


@njit
def pdf_last(multipdf_distribution, data, value):
    start = multipdf_distribution["pdf_offset"]
    size = multipdf_distribution["pdf_length"]
    end = start + size
    data[end - 1] = value


@njit
def pdf_chunk(start, length, multipdf_distribution, data, value):
    start += multipdf_distribution["pdf_offset"]
    end = start + length
    data[start:end] = value


@njit
def cdf(index, multipdf_distribution, data, value):
    offset = multipdf_distribution["cdf_offset"]
    data[offset + index] = value


@njit
def cdf_all(multipdf_distribution, data, value):
    start = multipdf_distribution["cdf_offset"]
    size = multipdf_distribution["cdf_length"]
    end = start + size
    data[start:end] = value


@njit
def cdf_last(multipdf_distribution, data, value):
    start = multipdf_distribution["cdf_offset"]
    size = multipdf_distribution["cdf_length"]
    end = start + size
    data[end - 1] = value


@njit
def cdf_chunk(start, length, multipdf_distribution, data, value):
    start += multipdf_distribution["cdf_offset"]
    end = start + length
    data[start:end] = value
