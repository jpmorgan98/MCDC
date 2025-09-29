from numba import njit


@njit
def x(index, table, data):
    offset = table["x_offset"]
    return data[offset + index]


@njit
def x_last(table, data):
    start = table["x_offset"]
    end = start + table["x_length"]
    return data[end - 1]


@njit
def x_all(table, data):
    start = table["x_offset"]
    end = start + table["x_length"]
    return data[start:end]


@njit
def x_chunk(start, length, table, data):
    start += table["x_offset"]
    end = start + length
    return data[start:end]


@njit
def y(index, table, data):
    offset = table["y_offset"]
    return data[offset + index]


@njit
def y_last(table, data):
    start = table["y_offset"]
    end = start + table["y_length"]
    return data[end - 1]


@njit
def y_all(table, data):
    start = table["y_offset"]
    end = start + table["y_length"]
    return data[start:end]


@njit
def y_chunk(start, length, table, data):
    start += table["y_offset"]
    end = start + length
    return data[start:end]
