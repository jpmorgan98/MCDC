from numba import njit


@njit
def delayed_yield_types(index, neutron_fission, data):
    offset = neutron_fission["delayed_yield_types_offset"]
    return data[offset + index]


@njit
def delayed_yield_types_last(neutron_fission, data):
    start = neutron_fission["delayed_yield_types_offset"]
    end = start + neutron_fission["delayed_yield_types_length"]
    return data[end - 1]


@njit
def delayed_yield_types_all(neutron_fission, data):
    start = neutron_fission["delayed_yield_types_offset"]
    end = start + neutron_fission["delayed_yield_types_length"]
    return data[start:end]


@njit
def delayed_yield_types_chunk(start, length, neutron_fission, data):
    start += neutron_fission["delayed_yield_types_offset"]
    end = start + length
    return data[start:end]


@njit
def delayed_yield_IDs(index, neutron_fission, data):
    offset = neutron_fission["delayed_yield_IDs_offset"]
    return data[offset + index]


@njit
def delayed_yield_IDs_last(neutron_fission, data):
    start = neutron_fission["delayed_yield_IDs_offset"]
    end = start + neutron_fission["delayed_yield_IDs_length"]
    return data[end - 1]


@njit
def delayed_yield_IDs_all(neutron_fission, data):
    start = neutron_fission["delayed_yield_IDs_offset"]
    end = start + neutron_fission["delayed_yield_IDs_length"]
    return data[start:end]


@njit
def delayed_yield_IDs_chunk(start, length, neutron_fission, data):
    start += neutron_fission["delayed_yield_IDs_offset"]
    end = start + length
    return data[start:end]


@njit
def delayed_spectrum_types(index, neutron_fission, data):
    offset = neutron_fission["delayed_spectrum_types_offset"]
    return data[offset + index]


@njit
def delayed_spectrum_types_last(neutron_fission, data):
    start = neutron_fission["delayed_spectrum_types_offset"]
    end = start + neutron_fission["delayed_spectrum_types_length"]
    return data[end - 1]


@njit
def delayed_spectrum_types_all(neutron_fission, data):
    start = neutron_fission["delayed_spectrum_types_offset"]
    end = start + neutron_fission["delayed_spectrum_types_length"]
    return data[start:end]


@njit
def delayed_spectrum_types_chunk(start, length, neutron_fission, data):
    start += neutron_fission["delayed_spectrum_types_offset"]
    end = start + length
    return data[start:end]


@njit
def delayed_spectrum_IDs(index, neutron_fission, data):
    offset = neutron_fission["delayed_spectrum_IDs_offset"]
    return data[offset + index]


@njit
def delayed_spectrum_IDs_last(neutron_fission, data):
    start = neutron_fission["delayed_spectrum_IDs_offset"]
    end = start + neutron_fission["delayed_spectrum_IDs_length"]
    return data[end - 1]


@njit
def delayed_spectrum_IDs_all(neutron_fission, data):
    start = neutron_fission["delayed_spectrum_IDs_offset"]
    end = start + neutron_fission["delayed_spectrum_IDs_length"]
    return data[start:end]


@njit
def delayed_spectrum_IDs_chunk(start, length, neutron_fission, data):
    start += neutron_fission["delayed_spectrum_IDs_offset"]
    end = start + length
    return data[start:end]


@njit
def delayed_decay_rates(index, neutron_fission, data):
    offset = neutron_fission["delayed_decay_rates_offset"]
    return data[offset + index]


@njit
def delayed_decay_rates_last(neutron_fission, data):
    start = neutron_fission["delayed_decay_rates_offset"]
    end = start + neutron_fission["delayed_decay_rates_length"]
    return data[end - 1]


@njit
def delayed_decay_rates_all(neutron_fission, data):
    start = neutron_fission["delayed_decay_rates_offset"]
    end = start + neutron_fission["delayed_decay_rates_length"]
    return data[start:end]


@njit
def delayed_decay_rates_chunk(start, length, neutron_fission, data):
    start += neutron_fission["delayed_decay_rates_offset"]
    end = start + length
    return data[start:end]
