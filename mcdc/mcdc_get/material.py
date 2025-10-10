from numba import njit


@njit
def nuclide_IDs(index, material, data):
    offset = material["nuclide_IDs_offset"]
    return data[offset + index]


@njit
def nuclide_IDs_all(material, data):
    start = material["nuclide_IDs_offset"]
    size = material["N_nuclide"]
    end = start + size
    return data[start:end]


@njit
def nuclide_IDs_last(material, data):
    start = material["nuclide_IDs_offset"]
    size = material["N_nuclide"]
    end = start + size
    return data[end - 1]


@njit
def nuclide_IDs_chunk(start, length, material, data):
    start += material["nuclide_IDs_offset"]
    end = start + length
    return data[start:end]


@njit
def nuclide_densities(index, material, data):
    offset = material["nuclide_densities_offset"]
    return data[offset + index]


@njit
def nuclide_densities_all(material, data):
    start = material["nuclide_densities_offset"]
    size = material["nuclide_densities_length"]
    end = start + size
    return data[start:end]


@njit
def nuclide_densities_last(material, data):
    start = material["nuclide_densities_offset"]
    size = material["nuclide_densities_length"]
    end = start + size
    return data[end - 1]


@njit
def nuclide_densities_chunk(start, length, material, data):
    start += material["nuclide_densities_offset"]
    end = start + length
    return data[start:end]
