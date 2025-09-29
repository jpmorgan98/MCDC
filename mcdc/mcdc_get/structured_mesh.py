from numba import njit


@njit
def x(index, structured_mesh, data):
    offset = structured_mesh["x_offset"]
    return data[offset + index]


@njit
def x_last(structured_mesh, data):
    start = structured_mesh["x_offset"]
    end = start + structured_mesh["x_length"]
    return data[end - 1]


@njit
def x_all(structured_mesh, data):
    start = structured_mesh["x_offset"]
    end = start + structured_mesh["x_length"]
    return data[start:end]


@njit
def x_chunk(start, length, structured_mesh, data):
    start += structured_mesh["x_offset"]
    end = start + length
    return data[start:end]


@njit
def y(index, structured_mesh, data):
    offset = structured_mesh["y_offset"]
    return data[offset + index]


@njit
def y_last(structured_mesh, data):
    start = structured_mesh["y_offset"]
    end = start + structured_mesh["y_length"]
    return data[end - 1]


@njit
def y_all(structured_mesh, data):
    start = structured_mesh["y_offset"]
    end = start + structured_mesh["y_length"]
    return data[start:end]


@njit
def y_chunk(start, length, structured_mesh, data):
    start += structured_mesh["y_offset"]
    end = start + length
    return data[start:end]


@njit
def z(index, structured_mesh, data):
    offset = structured_mesh["z_offset"]
    return data[offset + index]


@njit
def z_last(structured_mesh, data):
    start = structured_mesh["z_offset"]
    end = start + structured_mesh["z_length"]
    return data[end - 1]


@njit
def z_all(structured_mesh, data):
    start = structured_mesh["z_offset"]
    end = start + structured_mesh["z_length"]
    return data[start:end]


@njit
def z_chunk(start, length, structured_mesh, data):
    start += structured_mesh["z_offset"]
    end = start + length
    return data[start:end]


@njit
def t(index, structured_mesh, data):
    offset = structured_mesh["t_offset"]
    return data[offset + index]


@njit
def t_last(structured_mesh, data):
    start = structured_mesh["t_offset"]
    end = start + structured_mesh["t_length"]
    return data[end - 1]


@njit
def t_all(structured_mesh, data):
    start = structured_mesh["t_offset"]
    end = start + structured_mesh["t_length"]
    return data[start:end]


@njit
def t_chunk(start, length, structured_mesh, data):
    start += structured_mesh["t_offset"]
    end = start + length
    return data[start:end]
