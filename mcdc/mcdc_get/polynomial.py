from numba import njit


@njit
def coefficients_all(polynomial, data):
    start = polynomial["coefficients_offset"]
    end = start + polynomial["N_coefficients"]
    return data[start:end]


@njit
def coefficients(index, polynomial, data):
    offset = polynomial["coefficients_offset"]
    return data[offset + index]


@njit
def coefficients_chunk(start, size, polynomial, data):
    start += polynomial["coefficients_offset"]
    end = start + size
    return data[start:end]
