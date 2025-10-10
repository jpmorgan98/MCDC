from numba import njit


@njit
def value(index, pmf_data, data):
    offset = pmf_data["value_offset"]
    return data[offset + index]


@njit
def value_all(pmf_data, data):
    start = pmf_data["value_offset"]
    size = pmf_data["value_length"]
    end = start + size
    return data[start:end]


@njit
def value_last(pmf_data, data):
    start = pmf_data["value_offset"]
    size = pmf_data["value_length"]
    end = start + size
    return data[end - 1]


@njit
def value_chunk(start, length, pmf_data, data):
    start += pmf_data["value_offset"]
    end = start + length
    return data[start:end]


@njit
def pmf(index, pmf_data, data):
    offset = pmf_data["pmf_offset"]
    return data[offset + index]


@njit
def pmf_all(pmf_data, data):
    start = pmf_data["pmf_offset"]
    size = pmf_data["pmf_length"]
    end = start + size
    return data[start:end]


@njit
def pmf_last(pmf_data, data):
    start = pmf_data["pmf_offset"]
    size = pmf_data["pmf_length"]
    end = start + size
    return data[end - 1]


@njit
def pmf_chunk(start, length, pmf_data, data):
    start += pmf_data["pmf_offset"]
    end = start + length
    return data[start:end]


@njit
def cmf(index, pmf_data, data):
    offset = pmf_data["cmf_offset"]
    return data[offset + index]


@njit
def cmf_all(pmf_data, data):
    start = pmf_data["cmf_offset"]
    size = pmf_data["cmf_length"]
    end = start + size
    return data[start:end]


@njit
def cmf_last(pmf_data, data):
    start = pmf_data["cmf_offset"]
    size = pmf_data["cmf_length"]
    end = start + size
    return data[end - 1]


@njit
def cmf_chunk(start, length, pmf_data, data):
    start += pmf_data["cmf_offset"]
    end = start + length
    return data[start:end]
