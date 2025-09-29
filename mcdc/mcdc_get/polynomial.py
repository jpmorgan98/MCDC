from numba import njit


@njit
def coefficients(index, polynomial, data):
    offset = polynomial["coefficients_offset"]
    return data[offset + index]


@njit
def coefficients_last(polynomial, data):
    start = polynomial["coefficients_offset"]
    end = start + polynomial["coefficients_length"]
    return data[end - 1]


@njit
def coefficients_all(polynomial, data):
    start = polynomial["coefficients_offset"]
    end = start + polynomial["coefficients_length"]
    return data[start:end]


@njit
def coefficients_chunk(start, length, polynomial, data):
    start += polynomial["coefficients_offset"]
    end = start + length
    return data[start:end]
