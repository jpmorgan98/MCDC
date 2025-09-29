import math
from numba import njit

####

import mcdc.mcdc_get as mcdc_get

from mcdc.constant import (
    COINCIDENCE_TOLERANCE_DIRECTION,
    COINCIDENCE_TOLERANCE_ENERGY,
    COINCIDENCE_TOLERANCE_TIME,
)
from mcdc.transport.util import find_bin


@njit
def get_direction_index(particle_container, tally, data):
    particle = particle_container[0]

    # Particle properties
    ux = particle["ux"]
    uy = particle["uy"]
    uz = particle["uz"]

    # Polar reference
    nx = mcdc_get.tally.polar_reference(0, tally, data)
    ny = mcdc_get.tally.polar_reference(0, tally, data)
    nz = mcdc_get.tally.polar_reference(0, tally, data)

    # Rotate direction
    if nz != 1.0:
        # TODO
        pass

    mu = uz
    azi = math.acos(ux / math.sqrt(ux * ux + uy * uy))
    if uy < 0.0:
        azi *= -1

    tolerance = COINCIDENCE_TOLERANCE_DIRECTION
    i_mu = find_bin(mu, mcdc_get.tally.mu_all(tally, data), tolerance)
    i_azi = find_bin(azi, mcdc_get.tally.azi_all(tally, data), tolerance)
    return i_mu, i_azi


@njit
def get_energy_index(particle_container, tally, data, multigroup_mode):
    particle = particle_container[0]

    if multigroup_mode:
        E = particle["g"]
    else:
        E = particle["E"]

    tolerance = COINCIDENCE_TOLERANCE_ENERGY
    return find_bin(E, mcdc_get.tally.energy_all(tally, data), tolerance)


@njit
def get_time_index(particle_container, tally, data):
    particle = particle_container[0]

    # Particle properties
    time = particle["t"]

    tolerance = COINCIDENCE_TOLERANCE_TIME
    return find_bin(time, mcdc_get.tally.time_all(tally, data), tolerance)
