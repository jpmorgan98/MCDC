from numba import njit


@njit
def grid(index, multipdf_data, data):
    offset = multipdf_data["grid_offset"]
    return data[offset + index]


@njit
def grid_all(multipdf_data, data):
    start = multipdf_data["grid_offset"]
    size = multipdf_data["grid_length"]
    end = start + size
    return data[start:end]


@njit
def grid_last(multipdf_data, data):
    start = multipdf_data["grid_offset"]
    size = multipdf_data["grid_length"]
    end = start + size
    return data[end - 1]


@njit
def grid_chunk(start, length, multipdf_data, data):
    start += multipdf_data["grid_offset"]
    end = start + length
    return data[start:end]


@njit
def offset(index, multipdf_data, data):
    offset = multipdf_data["offset_offset"]
    return data[offset + index]


@njit
def offset_all(multipdf_data, data):
    start = multipdf_data["offset_offset"]
    size = multipdf_data["offset_length"]
    end = start + size
    return data[start:end]


@njit
def offset_last(multipdf_data, data):
    start = multipdf_data["offset_offset"]
    size = multipdf_data["offset_length"]
    end = start + size
    return data[end - 1]


@njit
def offset_chunk(start, length, multipdf_data, data):
    start += multipdf_data["offset_offset"]
    end = start + length
    return data[start:end]


@njit
def value(index, multipdf_data, data):
    offset = multipdf_data["value_offset"]
    return data[offset + index]


@njit
def value_all(multipdf_data, data):
    start = multipdf_data["value_offset"]
    size = multipdf_data["value_length"]
    end = start + size
    return data[start:end]


@njit
def value_last(multipdf_data, data):
    start = multipdf_data["value_offset"]
    size = multipdf_data["value_length"]
    end = start + size
    return data[end - 1]


@njit
def value_chunk(start, length, multipdf_data, data):
    start += multipdf_data["value_offset"]
    end = start + length
    return data[start:end]


@njit
def pdf(index, multipdf_data, data):
    offset = multipdf_data["pdf_offset"]
    return data[offset + index]


@njit
def pdf_all(multipdf_data, data):
    start = multipdf_data["pdf_offset"]
    size = multipdf_data["pdf_length"]
    end = start + size
    return data[start:end]


@njit
def pdf_last(multipdf_data, data):
    start = multipdf_data["pdf_offset"]
    size = multipdf_data["pdf_length"]
    end = start + size
    return data[end - 1]


@njit
def pdf_chunk(start, length, multipdf_data, data):
    start += multipdf_data["pdf_offset"]
    end = start + length
    return data[start:end]


@njit
def cdf(index, multipdf_data, data):
    offset = multipdf_data["cdf_offset"]
    return data[offset + index]


@njit
def cdf_all(multipdf_data, data):
    start = multipdf_data["cdf_offset"]
    size = multipdf_data["cdf_length"]
    end = start + size
    return data[start:end]


@njit
def cdf_last(multipdf_data, data):
    start = multipdf_data["cdf_offset"]
    size = multipdf_data["cdf_length"]
    end = start + size
    return data[end - 1]


@njit
def cdf_chunk(start, length, multipdf_data, data):
    start += multipdf_data["cdf_offset"]
    end = start + length
    return data[start:end]
