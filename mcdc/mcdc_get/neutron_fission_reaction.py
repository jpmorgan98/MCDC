from numba import njit


@njit
def delayed_yield_IDs(index, neutron_fission_reaction, data):
    offset = neutron_fission_reaction["delayed_yield_IDs_offset"]
    return data[offset + index]


@njit
def delayed_yield_IDs_all(neutron_fission_reaction, data):
    start = neutron_fission_reaction["delayed_yield_IDs_offset"]
    size = neutron_fission_reaction["N_delayed_yield"]
    end = start + size
    return data[start:end]


@njit
def delayed_yield_IDs_last(neutron_fission_reaction, data):
    start = neutron_fission_reaction["delayed_yield_IDs_offset"]
    size = neutron_fission_reaction["N_delayed_yield"]
    end = start + size
    return data[end - 1]


@njit
def delayed_yield_IDs_chunk(start, length, neutron_fission_reaction, data):
    start += neutron_fission_reaction["delayed_yield_IDs_offset"]
    end = start + length
    return data[start:end]


@njit
def delayed_yield_types(index, neutron_fission_reaction, data):
    offset = neutron_fission_reaction["delayed_yield_types_offset"]
    return data[offset + index]


@njit
def delayed_yield_types_all(neutron_fission_reaction, data):
    start = neutron_fission_reaction["delayed_yield_types_offset"]
    size = neutron_fission_reaction["N_delayed_yield"]
    end = start + size
    return data[start:end]


@njit
def delayed_yield_types_last(neutron_fission_reaction, data):
    start = neutron_fission_reaction["delayed_yield_types_offset"]
    size = neutron_fission_reaction["N_delayed_yield"]
    end = start + size
    return data[end - 1]


@njit
def delayed_yield_types_chunk(start, length, neutron_fission_reaction, data):
    start += neutron_fission_reaction["delayed_yield_types_offset"]
    end = start + length
    return data[start:end]


@njit
def delayed_spectrum_IDs(index, neutron_fission_reaction, data):
    offset = neutron_fission_reaction["delayed_spectrum_IDs_offset"]
    return data[offset + index]


@njit
def delayed_spectrum_IDs_all(neutron_fission_reaction, data):
    start = neutron_fission_reaction["delayed_spectrum_IDs_offset"]
    size = neutron_fission_reaction["N_delayed_spectrum"]
    end = start + size
    return data[start:end]


@njit
def delayed_spectrum_IDs_last(neutron_fission_reaction, data):
    start = neutron_fission_reaction["delayed_spectrum_IDs_offset"]
    size = neutron_fission_reaction["N_delayed_spectrum"]
    end = start + size
    return data[end - 1]


@njit
def delayed_spectrum_IDs_chunk(start, length, neutron_fission_reaction, data):
    start += neutron_fission_reaction["delayed_spectrum_IDs_offset"]
    end = start + length
    return data[start:end]


@njit
def delayed_spectrum_types(index, neutron_fission_reaction, data):
    offset = neutron_fission_reaction["delayed_spectrum_types_offset"]
    return data[offset + index]


@njit
def delayed_spectrum_types_all(neutron_fission_reaction, data):
    start = neutron_fission_reaction["delayed_spectrum_types_offset"]
    size = neutron_fission_reaction["N_delayed_spectrum"]
    end = start + size
    return data[start:end]


@njit
def delayed_spectrum_types_last(neutron_fission_reaction, data):
    start = neutron_fission_reaction["delayed_spectrum_types_offset"]
    size = neutron_fission_reaction["N_delayed_spectrum"]
    end = start + size
    return data[end - 1]


@njit
def delayed_spectrum_types_chunk(start, length, neutron_fission_reaction, data):
    start += neutron_fission_reaction["delayed_spectrum_types_offset"]
    end = start + length
    return data[start:end]


@njit
def delayed_decay_rates(index, neutron_fission_reaction, data):
    offset = neutron_fission_reaction["delayed_decay_rates_offset"]
    return data[offset + index]


@njit
def delayed_decay_rates_all(neutron_fission_reaction, data):
    start = neutron_fission_reaction["delayed_decay_rates_offset"]
    size = neutron_fission_reaction["delayed_decay_rates_length"]
    end = start + size
    return data[start:end]


@njit
def delayed_decay_rates_last(neutron_fission_reaction, data):
    start = neutron_fission_reaction["delayed_decay_rates_offset"]
    size = neutron_fission_reaction["delayed_decay_rates_length"]
    end = start + size
    return data[end - 1]


@njit
def delayed_decay_rates_chunk(start, length, neutron_fission_reaction, data):
    start += neutron_fission_reaction["delayed_decay_rates_offset"]
    end = start + length
    return data[start:end]


@njit
def xs(index, neutron_fission_reaction, data):
    offset = neutron_fission_reaction["xs_offset"]
    return data[offset + index]


@njit
def xs_all(neutron_fission_reaction, data):
    start = neutron_fission_reaction["xs_offset"]
    size = neutron_fission_reaction["xs_length"]
    end = start + size
    return data[start:end]


@njit
def xs_last(neutron_fission_reaction, data):
    start = neutron_fission_reaction["xs_offset"]
    size = neutron_fission_reaction["xs_length"]
    end = start + size
    return data[end - 1]


@njit
def xs_chunk(start, length, neutron_fission_reaction, data):
    start += neutron_fission_reaction["xs_offset"]
    end = start + length
    return data[start:end]
