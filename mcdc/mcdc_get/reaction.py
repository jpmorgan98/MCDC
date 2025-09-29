from numba import njit


@njit
def xs(index, reaction, data):
    offset = reaction["xs_offset"]
    return data[offset + index]


@njit
def xs_last(reaction, data):
    start = reaction["xs_offset"]
    end = start + reaction["xs_length"]
    return data[end - 1]


@njit
def xs_all(reaction, data):
    start = reaction["xs_offset"]
    end = start + reaction["xs_length"]
    return data[start:end]


@njit
def xs_chunk(start, length, reaction, data):
    start += reaction["xs_offset"]
    end = start + length
    return data[start:end]
