from numba import njit


@njit
def move_velocities(index, surface, data):
    offset = surface["move_velocities_offset"]
    return data[offset + index]


@njit
def move_velocities_all(surface, data):
    start = surface["move_velocities_offset"]
    size = surface["move_velocities_length"]
    end = start + size
    return data[start:end]


@njit
def move_velocities_last(surface, data):
    start = surface["move_velocities_offset"]
    size = surface["move_velocities_length"]
    end = start + size
    return data[end - 1]


@njit
def move_velocities_chunk(start, length, surface, data):
    start += surface["move_velocities_offset"]
    end = start + length
    return data[start:end]


@njit
def move_durations(index, surface, data):
    offset = surface["move_durations_offset"]
    return data[offset + index]


@njit
def move_durations_all(surface, data):
    start = surface["move_durations_offset"]
    size = surface["move_durations_length"]
    end = start + size
    return data[start:end]


@njit
def move_durations_last(surface, data):
    start = surface["move_durations_offset"]
    size = surface["move_durations_length"]
    end = start + size
    return data[end - 1]


@njit
def move_durations_chunk(start, length, surface, data):
    start += surface["move_durations_offset"]
    end = start + length
    return data[start:end]


@njit
def move_time_grid(index, surface, data):
    offset = surface["move_time_grid_offset"]
    return data[offset + index]


@njit
def move_time_grid_all(surface, data):
    start = surface["move_time_grid_offset"]
    size = surface["move_time_grid_length"]
    end = start + size
    return data[start:end]


@njit
def move_time_grid_last(surface, data):
    start = surface["move_time_grid_offset"]
    size = surface["move_time_grid_length"]
    end = start + size
    return data[end - 1]


@njit
def move_time_grid_chunk(start, length, surface, data):
    start += surface["move_time_grid_offset"]
    end = start + length
    return data[start:end]


@njit
def move_translations(index, surface, data):
    offset = surface["move_translations_offset"]
    return data[offset + index]


@njit
def move_translations_all(surface, data):
    start = surface["move_translations_offset"]
    size = surface["move_translations_length"]
    end = start + size
    return data[start:end]


@njit
def move_translations_last(surface, data):
    start = surface["move_translations_offset"]
    size = surface["move_translations_length"]
    end = start + size
    return data[end - 1]


@njit
def move_translations_chunk(start, length, surface, data):
    start += surface["move_translations_offset"]
    end = start + length
    return data[start:end]
