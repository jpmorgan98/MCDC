from numba import njit


@njit
def value(index, pdf_data, data):
    offset = pdf_data["value_offset"]
    return data[offset + index]


@njit
def value_all(pdf_data, data):
    start = pdf_data["value_offset"]
    size = pdf_data["value_length"]
    end = start + size
    return data[start:end]


@njit
def value_last(pdf_data, data):
    start = pdf_data["value_offset"]
    size = pdf_data["value_length"]
    end = start + size
    return data[end - 1]


@njit
def value_chunk(start, length, pdf_data, data):
    start += pdf_data["value_offset"]
    end = start + length
    return data[start:end]


@njit
def pdf(index, pdf_data, data):
    offset = pdf_data["pdf_offset"]
    return data[offset + index]


@njit
def pdf_all(pdf_data, data):
    start = pdf_data["pdf_offset"]
    size = pdf_data["pdf_length"]
    end = start + size
    return data[start:end]


@njit
def pdf_last(pdf_data, data):
    start = pdf_data["pdf_offset"]
    size = pdf_data["pdf_length"]
    end = start + size
    return data[end - 1]


@njit
def pdf_chunk(start, length, pdf_data, data):
    start += pdf_data["pdf_offset"]
    end = start + length
    return data[start:end]


@njit
def cdf(index, pdf_data, data):
    offset = pdf_data["cdf_offset"]
    return data[offset + index]


@njit
def cdf_all(pdf_data, data):
    start = pdf_data["cdf_offset"]
    size = pdf_data["cdf_length"]
    end = start + size
    return data[start:end]


@njit
def cdf_last(pdf_data, data):
    start = pdf_data["cdf_offset"]
    size = pdf_data["cdf_length"]
    end = start + size
    return data[end - 1]


@njit
def cdf_chunk(start, length, pdf_data, data):
    start += pdf_data["cdf_offset"]
    end = start + length
    return data[start:end]
