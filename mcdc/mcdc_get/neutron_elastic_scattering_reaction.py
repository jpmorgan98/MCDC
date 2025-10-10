from numba import njit


@njit
def xs(index, neutron_elastic_scattering_reaction, data):
    offset = neutron_elastic_scattering_reaction["xs_offset"]
    return data[offset + index]


@njit
def xs_all(neutron_elastic_scattering_reaction, data):
    start = neutron_elastic_scattering_reaction["xs_offset"]
    size = neutron_elastic_scattering_reaction["xs_length"]
    end = start + size
    return data[start:end]


@njit
def xs_last(neutron_elastic_scattering_reaction, data):
    start = neutron_elastic_scattering_reaction["xs_offset"]
    size = neutron_elastic_scattering_reaction["xs_length"]
    end = start + size
    return data[end - 1]


@njit
def xs_chunk(start, length, neutron_elastic_scattering_reaction, data):
    start += neutron_elastic_scattering_reaction["xs_offset"]
    end = start + length
    return data[start:end]
