from numba import njit


@njit
def region_RPN_tokens_all(cell, data):
    start = cell["region_RPN_tokens_offset"]
    end = start + cell["N_region_RPN_tokens"]
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


@njit
def surface_index_all(cell, data):
    start = cell["surface_index_offset"]
    end = start + cell["N_surface_index"]
    return data[start:end]


@njit
def surface_index(index, cell, data):
    offset = cell["surface_index_offset"]
    return data[offset + index]


@njit
def surface_index_chunk(start, size, cell, data):
    start += cell["surface_index_offset"]
    end = start + size
    return data[start:end]


@njit
def translation_all(cell, data):
    start = cell["translation_offset"]
    end = start + cell["N_translation"]
    return data[start:end]


@njit
def translation(index, cell, data):
    offset = cell["translation_offset"]
    return data[offset + index]


@njit
def translation_chunk(start, size, cell, data):
    start += cell["translation_offset"]
    end = start + size
    return data[start:end]


@njit
def rotation_all(cell, data):
    start = cell["rotation_offset"]
    end = start + cell["N_rotation"]
    return data[start:end]


@njit
def rotation(index, cell, data):
    offset = cell["rotation_offset"]
    return data[offset + index]


@njit
def rotation_chunk(start, size, cell, data):
    start += cell["rotation_offset"]
    end = start + size
    return data[start:end]
