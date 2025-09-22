from numba import njit


@njit
def region_RPN_tokens_length(cell):
    return int(cell["region_RPN_tokens_length"])


@njit
def region_RPN_tokens_all(cell, data):
    start = cell["region_RPN_tokens_offset"]
    end = start + cell["region_RPN_tokens_length"]
    return data[start:end]


@njit
def region_RPN_tokens(index, cell, data):
    offset = cell["region_RPN_tokens_offset"]
    return data[offset + index]


@njit
def region_RPN_tokens_chunk(start, size, cell, data):
    start += cell["region_RPN_tokens_offset"]
    end = start + size
    return data[start:end]
