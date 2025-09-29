import math
import numpy as np

from numba import njit, objmode
from mpi4py import MPI

from mcdc.print_ import print_structure


# ======================================================================================
# Reduce tally bins
# ======================================================================================

@njit
def reduce(mcdc, data):
    for tally in mcdc['cell_tallys']:
        _reduce(tally, mcdc, data)
    #for tally in mcdc['surface_tallys']:
    #    _reduce(tally, mcdc, data)
    for tally in mcdc['mesh_tallys']:
        _reduce(tally, mcdc, data)

@njit
def _reduce(tally, mcdc, data):
    N = tally['bin_length']
    start = tally['bin_offset']
    end = start + N

    # Normalize
    N_particle = mcdc["settings"]["N_particle"]
    for i in range(N):
        data[start + i] /= N_particle

    # Reduce
    if not mcdc["technique"]["domain_decomposition"]:
        # MPI Reduce
        buff = np.zeros(N)
        with objmode():
            MPI.COMM_WORLD.Reduce(data[start:end], buff, MPI.SUM, 0)
        data[start:end] = buff

    else:
        # find number of subdomains
        N_dd = 1
        N_dd *= mcdc["technique"]["dd_mesh"]["x"].size - 1
        N_dd *= mcdc["technique"]["dd_mesh"]["y"].size - 1
        N_dd *= mcdc["technique"]["dd_mesh"]["z"].size - 1
        # DD Reduce if multiple processors per subdomain
        if N_dd != mcdc["mpi_size"]:
            pass
            # TODO: dd_reduce(data, mcdc)


# ======================================================================================
# Accumulate tally bins
# ======================================================================================


@njit
def accumulate(mcdc, data):
    for tally in mcdc['cell_tallys']:
        _accumulate(tally, mcdc, data)
    #for tally in mcdc['surface_tallys']:
    #    _accumulate(tally, mcdc, data)
    for tally in mcdc['mesh_tallys']:
        _accumulate(tally, mcdc, data)


@njit
def _accumulate(tally, mcdc, data):
    N_bin = tally['bin_length']
    offset_bin = tally['bin_offset']
    offset_sum = tally['bin_sum_offset']
    offset_sum_square = tally['bin_sum_square_offset']

    for i in range(N_bin):
        # Accumulate score and square of score into sum and sum_sq
        score = data[offset_bin + i]
        data[offset_sum + i] += score
        data[offset_sum_square + i] += score * score

        # Reset score bin
        data[offset_bin + i] = 0.0


# ======================================================================================
# Finalize
# ======================================================================================


@njit
def finalize(mcdc, data):
    for tally in mcdc['cell_tallys']:
        _finalize(tally, mcdc, data)
    #for tally in mcdc['surface_tallys']:
    #    _finalize(tally, mcdc, data)
    for tally in mcdc['mesh_tallys']:
        _finalize(tally, mcdc, data)


@njit
def _finalize(tally, mcdc, data):
    N_history = mcdc["settings"]["N_particle"]
    N_batch = mcdc["settings"]["N_batch"]

    if N_batch > 1:
        N_history = N_batch

    elif mcdc["settings"]["eigenvalue_mode"]:
        N_history = mcdc["settings"]["N_active"]

    elif not mcdc["technique"]["domain_decomposition"]:
        # MPI Reduce
        buff = np.zeros_like(data[TALLY_SUM])
        buff_sq = np.zeros_like(data[TALLY_SUM_SQ])
        with objmode():
            MPI.COMM_WORLD.Reduce(data[TALLY_SUM], buff, MPI.SUM, 0)
            MPI.COMM_WORLD.Reduce(data[TALLY_SUM_SQ], buff_sq, MPI.SUM, 0)
        data[TALLY_SUM] = buff
        data[TALLY_SUM_SQ] = buff_sq

    else:
        # find number of subdomains
        N_dd = 1
        N_dd *= mcdc["technique"]["dd_mesh"]["x"].size - 1
        N_dd *= mcdc["technique"]["dd_mesh"]["y"].size - 1
        N_dd *= mcdc["technique"]["dd_mesh"]["z"].size - 1
        # DD Reduce if multiple processors per subdomain
        if N_dd != mcdc["mpi_size"]:
            dd_closeout(data, mcdc)

    # Calculate and store statistics
    #   sum --> mean
    #   sum_sq --> standard deviation
    N_bin = tally['bin_length']
    offset_sum = tally['bin_sum_offset']
    offset_sum_square = tally['bin_sum_square_offset']
    for i in range(N_bin):
        data[offset_sum + i] = data[offset_sum + i] / N_history
        radicand = (
            data[offset_sum_square + i] / N_history - np.square(data[offset_sum + i])
        ) / (N_history - 1)

        # Check for round-off error
        if abs(radicand) < 1e-18:
            data[offset_sum_square + i] = 0.0
        else:
            data[offset_sum_square + i] = math.sqrt(radicand)


