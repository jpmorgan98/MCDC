from numba import njit


@njit
def delayed_yield_type_all(neutron_fission, data):
    start = neutron_fission["delayed_yield_type_offset"]
    end = start + neutron_fission["N_delayed_yield_type"]
    return data[start:end]


@njit
def delayed_yield_type(index, neutron_fission, data):
    offset = neutron_fission["delayed_yield_type_offset"]
    return data[offset + index]


@njit
def delayed_yield_type_chunk(start, size, neutron_fission, data):
    start += neutron_fission["delayed_yield_type_offset"]
    end = start + size
    return data[start:end]


@njit
def delayed_yield_index_all(neutron_fission, data):
    start = neutron_fission["delayed_yield_index_offset"]
    end = start + neutron_fission["N_delayed_yield_index"]
    return data[start:end]


@njit
def delayed_yield_index(index, neutron_fission, data):
    offset = neutron_fission["delayed_yield_index_offset"]
    return data[offset + index]


@njit
def delayed_yield_index_chunk(start, size, neutron_fission, data):
    start += neutron_fission["delayed_yield_index_offset"]
    end = start + size
    return data[start:end]


@njit
def delayed_spectrum_type_all(neutron_fission, data):
    start = neutron_fission["delayed_spectrum_type_offset"]
    end = start + neutron_fission["N_delayed_spectrum_type"]
    return data[start:end]


@njit
def delayed_spectrum_type(index, neutron_fission, data):
    offset = neutron_fission["delayed_spectrum_type_offset"]
    return data[offset + index]


@njit
def delayed_spectrum_type_chunk(start, size, neutron_fission, data):
    start += neutron_fission["delayed_spectrum_type_offset"]
    end = start + size
    return data[start:end]


@njit
def delayed_spectrum_index_all(neutron_fission, data):
    start = neutron_fission["delayed_spectrum_index_offset"]
    end = start + neutron_fission["N_delayed_spectrum_index"]
    return data[start:end]


@njit
def delayed_spectrum_index(index, neutron_fission, data):
    offset = neutron_fission["delayed_spectrum_index_offset"]
    return data[offset + index]


@njit
def delayed_spectrum_index_chunk(start, size, neutron_fission, data):
    start += neutron_fission["delayed_spectrum_index_offset"]
    end = start + size
    return data[start:end]


@njit
def delayed_decay_rates_all(neutron_fission, data):
    start = neutron_fission["delayed_decay_rates_offset"]
    end = start + neutron_fission["N_delayed_decay_rates"]
    return data[start:end]


@njit
def delayed_decay_rates(index, neutron_fission, data):
    offset = neutron_fission["delayed_decay_rates_offset"]
    return data[offset + index]


@njit
def delayed_decay_rates_chunk(start, size, neutron_fission, data):
    start += neutron_fission["delayed_decay_rates_offset"]
    end = start + size
    return data[start:end]
