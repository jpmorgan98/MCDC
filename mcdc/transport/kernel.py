import mcdc.mcdc_get as mcdc_get

import h5py, math
import numpy as np

from mpi4py import MPI
from numba import (
    literal_unroll,
    njit,
    objmode,
    uint64,
)

####

import mcdc.transport.mpi as mpi
import mcdc.transport.physics as physics
import mcdc.transport.geometry as geometry
import mcdc.transport.technique as technique

import mcdc.code_factory.adapt as adapt
import mcdc.transport.mesh as mesh_

import mcdc.transport.tally as tally_module
import mcdc.transport.rng as rng
import mcdc.object_.numba_types as type_

from mcdc.code_factory.adapt import toggle, for_cpu, for_gpu
from mcdc.constant import *
from mcdc.print_ import print_error, print_structure

import cffi

ffi = cffi.FFI()


# =============================================================================
# Particle bank operations
# =============================================================================


@njit
def get_bank_size(bank):
    return bank["size"][0]


@njit
def set_bank_size(bank, value):
    bank["size"][0] = value


@njit
def add_bank_size(bank, value):
    return adapt.global_add(bank["size"], 0, value)


@for_cpu()
def full_bank_print(bank):
    with objmode():
        print_error(
            "Particle %s bank is full at count %d." % (bank["tag"], bank["size"])
        )


@for_gpu()
def full_bank_print(bank):
    pass


@njit
def add_full_particle(P_arr, bank):
    P = P_arr[0]

    idx = add_bank_size(bank, 1)

    # Check if bank is full
    if idx >= bank["particles"].shape[0]:
        full_bank_print(bank)

    # Set particle
    copy_particle(bank["particles"][idx : idx + 1], P_arr)


@njit
def add_particle(P_arr, bank):
    P = P_arr[0]

    idx = add_bank_size(bank, 1)

    # Check if bank is full
    if idx >= bank["particles"].shape[0]:
        full_bank_print(bank)

    # Set particle
    copy_recordlike(bank["particles"][idx : idx + 1], P_arr)


@njit
def get_particle(P_arr, bank, mcdc):
    P = P_arr[0]

    idx = add_bank_size(bank, -1) - 1

    # Check if bank is empty
    if idx < 0:
        return False
        # with objmode():
        #    print_error("Particle %s bank is empty." % bank["tag"])

    # Set attribute
    P_rec = bank["particles"][idx]
    P["x"] = P_rec["x"]
    P["y"] = P_rec["y"]
    P["z"] = P_rec["z"]
    P["t"] = P_rec["t"]
    P["ux"] = P_rec["ux"]
    P["uy"] = P_rec["uy"]
    P["uz"] = P_rec["uz"]
    P["g"] = P_rec["g"]
    P["E"] = P_rec["E"]
    P["w"] = P_rec["w"]
    P["particle_type"] = P_rec["particle_type"]
    P["rng_seed"] = P_rec["rng_seed"]

    # Set default IDs and event
    P["alive"] = True
    P["material_ID"] = -1
    P["cell_ID"] = -1
    P["surface_ID"] = -1
    P["event"] = -1
    return True


@njit
def check_future_bank(mcdc, data):
    # Get the data needed
    settings = mcdc["settings"]
    bank_future = mcdc["bank_future"]
    bank_census = mcdc["bank_census"]
    next_census_time = mcdc_get.settings.census_time(
        mcdc["idx_census"] + 1, settings, data
    )

    # Particle container
    P_arr = np.zeros(1, type_.particle_data)
    P = P_arr[0]

    # Loop over all particles in future bank
    N = get_bank_size(bank_future)
    for i in range(N):
        # Get the next future particle index
        idx = i - get_bank_size(bank_census)
        copy_recordlike(P_arr, bank_future["particles"][idx : idx + 1])

        # Promote the future particle to census bank
        if P["t"] < next_census_time:
            adapt.add_census(P_arr, mcdc)
            add_bank_size(bank_future, -1)

            # Consolidate the emptied space in the future bank
            j = get_bank_size(bank_future)
            copy_recordlike(
                bank_future["particles"][idx : idx + 1],
                bank_future["particles"][j : j + 1],
            )


@njit
def manage_particle_banks(mcdc):
    # Record time
    if mcdc["mpi_master"]:
        with objmode(time_start="float64"):
            time_start = MPI.Wtime()

    # Reset source bank
    set_bank_size(mcdc["bank_source"], 0)

    # Normalize weight
    if mcdc["settings"]["eigenvalue_mode"]:
        normalize_weight(mcdc["bank_census"], mcdc["settings"]["N_particle"])

    # Population control
    if mcdc["population_control"]["active"]:
        technique.population_control(mcdc)
    else:
        # Swap census and source bank
        size = get_bank_size(mcdc["bank_census"])
        set_bank_size(mcdc["bank_source"], size)
        mcdc["bank_source"]["particles"][:size] = mcdc["bank_census"]["particles"][
            :size
        ]
    # TODO: Population control future bank?

    # MPI rebalance
    bank_rebalance(mcdc)

    # Zero out census bank
    set_bank_size(mcdc["bank_census"], 0)

    # Accumulate time
    if mcdc["mpi_master"]:
        with objmode(time_end="float64"):
            time_end = MPI.Wtime()
        mcdc["runtime_bank_management"] += time_end - time_start


@njit
def bank_scanning(bank, mcdc):
    N_local = get_bank_size(bank)

    # Starting index
    buff = np.zeros(1, dtype=np.int64)
    with objmode():
        MPI.COMM_WORLD.Exscan(np.array([N_local]), buff, MPI.SUM)
    idx_start = buff[0]

    # Global size
    buff[0] += N_local
    with objmode():
        MPI.COMM_WORLD.Bcast(buff, mcdc["mpi_size"] - 1)
    N_global = buff[0]

    return idx_start, N_local, N_global


@njit
def bank_scanning_weight(bank, mcdc):
    # Local weight CDF
    N_local = get_bank_size(bank)
    w_cdf = np.zeros(N_local + 1)
    for i in range(N_local):
        w_cdf[i + 1] = w_cdf[i] + bank["particles"][i]["w"]
    W_local = w_cdf[-1]

    # Starting weight
    buff = np.zeros(1, dtype=np.float64)
    with objmode():
        MPI.COMM_WORLD.Exscan(np.array([W_local]), buff, MPI.SUM)
    w_start = buff[0]
    w_cdf += w_start

    # Global weight
    buff[0] = w_cdf[-1]
    with objmode():
        MPI.COMM_WORLD.Bcast(buff, mcdc["mpi_size"] - 1)
    W_global = buff[0]

    return w_start, w_cdf, W_global


@njit
def normalize_weight(bank, norm):
    # Get total weight
    W = total_weight(bank)

    # Normalize weight
    for i in range(get_bank_size(bank)):
        bank["particles"][i]["w"] *= norm / W


@njit
def total_weight(bank):
    # Local total weight
    W_local = np.zeros(1)
    for i in range(get_bank_size(bank)):
        W_local[0] += bank["particles"][i]["w"]

    # MPI Allreduce
    buff = np.zeros(1, np.float64)
    with objmode():
        MPI.COMM_WORLD.Allreduce(W_local, buff, MPI.SUM)
    return buff[0]


@njit
def allreduce(value):
    total = np.zeros(1, np.float64)
    with objmode():
        MPI.COMM_WORLD.Allreduce(np.array([value], np.float64), total, MPI.SUM)
    return total[0]


@njit
def allreduce_array(array):
    buff = np.zeros_like(array)
    with objmode():
        MPI.COMM_WORLD.Allreduce(np.array(array), buff, op=MPI.SUM)
    array[:] = buff


@njit
def bank_rebalance(mcdc):
    # Scan the bank
    idx_start, N_local, N = bank_scanning(mcdc["bank_source"], mcdc)
    idx_end = idx_start + N_local

    # Abort if source bank is empty
    if N == 0:
        return

    mpi.distribute_work(N, mcdc)

    # Rebalance not needed if there is only one rank
    if mcdc["mpi_size"] <= 1:
        return

    # Some constants
    work_start = mcdc["mpi_work_start"]
    work_end = work_start + mcdc["mpi_work_size"]
    left = mcdc["mpi_rank"] - 1
    right = mcdc["mpi_rank"] + 1

    # Need more or less?
    more_left = idx_start < work_start
    less_left = idx_start > work_start
    more_right = idx_end > work_end
    less_right = idx_end < work_end

    # Offside?
    offside_left = idx_end <= work_start
    offside_right = idx_start >= work_end

    # MPI nearest-neighbor send/receive
    buff = np.zeros(
        mcdc["bank_source"]["particles"].shape[0], dtype=type_.particle_data
    )

    with objmode(size="int64"):
        # Create MPI-supported numpy object
        size = get_bank_size(mcdc["bank_source"])
        bank = np.array(mcdc["bank_source"]["particles"][:size])

        # If offside, need to receive first
        if offside_left:
            # Receive from right
            bank = np.append(bank, MPI.COMM_WORLD.recv(source=right))
            less_right = False
        if offside_right:
            # Receive from left
            bank = np.insert(bank, 0, MPI.COMM_WORLD.recv(source=left))
            less_left = False

        # Send
        if more_left:
            n = work_start - idx_start
            request_left = MPI.COMM_WORLD.isend(bank[:n], dest=left)
            bank = bank[n:]
        if more_right:
            n = idx_end - work_end
            request_right = MPI.COMM_WORLD.isend(bank[-n:], dest=right)
            bank = bank[:-n]

        # Receive
        if less_left:
            bank = np.insert(bank, 0, MPI.COMM_WORLD.recv(source=left))
        if less_right:
            bank = np.append(bank, MPI.COMM_WORLD.recv(source=right))

        # Wait until sent massage is received
        if more_left:
            request_left.Wait()
        if more_right:
            request_right.Wait()

        # Set output buffer
        size = bank.shape[0]
        for i in range(size):
            buff[i] = bank[i]

    # Set source bank from buffer
    set_bank_size(mcdc["bank_source"], size)
    for i in range(size):
        mcdc["bank_source"]["particles"][i] = buff[i]


# =============================================================================
# Particle operations
# =============================================================================


@njit
def move_particle(P_arr, distance, mcdc, data):
    P = P_arr[0]
    P["x"] += P["ux"] * distance
    P["y"] += P["uy"] * distance
    P["z"] += P["uz"] * distance
    P["t"] += distance / physics.particle_speed(P_arr, mcdc, data)


@njit
def copy_recordlike(P_new_arr, P_rec_arr):
    P_new = P_new_arr[0]
    P_rec = P_rec_arr[0]
    P_new["x"] = P_rec["x"]
    P_new["y"] = P_rec["y"]
    P_new["z"] = P_rec["z"]
    P_new["t"] = P_rec["t"]
    P_new["ux"] = P_rec["ux"]
    P_new["uy"] = P_rec["uy"]
    P_new["uz"] = P_rec["uz"]
    P_new["g"] = P_rec["g"]
    P_new["E"] = P_rec["E"]
    P_new["w"] = P_rec["w"]
    P_new["particle_type"] = P_rec["particle_type"]
    P_new["rng_seed"] = P_rec["rng_seed"]


@njit
def copy_particle(P_new_arr, P_arr):
    P_new = P_new_arr[0]
    P = P_arr[0]
    P_new = P_new_arr[0]
    P_new["x"] = P["x"]
    P_new["y"] = P["y"]
    P_new["z"] = P["z"]
    P_new["t"] = P["t"]
    P_new["ux"] = P["ux"]
    P_new["uy"] = P["uy"]
    P_new["uz"] = P["uz"]
    P_new["g"] = P["g"]
    P_new["w"] = P["w"]
    P_new["type"] = P["type"]
    P_new["alive"] = P["alive"]
    P_new["fresh"] = P["fresh"]
    P_new["material_ID"] = P["material_ID"]
    P_new["cell_ID"] = P["cell_ID"]
    P_new["surface_ID"] = P["surface_ID"]
    P_new["event"] = P["event"]
    P_new["rng_seed"] = P["rng_seed"]


@njit
def recordlike_to_particle(P_new_arr, P_rec_arr):
    P_new = P_new_arr[0]
    P_rec = P_rec_arr[0]
    copy_recordlike(P_new_arr, P_rec_arr)
    P_new["fresh"] = True
    P_new["alive"] = True
    P_new["material_ID"] = -1
    P_new["cell_ID"] = -1
    P_new["surface_ID"] = -1
    P_new["event"] = -1


@njit
def split_as_data(P_new_rec_arr, P_rec_arr):
    P_rec = P_rec_arr[0]
    P_new_rec = P_new_rec_arr[0]
    copy_recordlike(P_new_rec_arr, P_rec_arr)
    P_new_rec["rng_seed"] = rng.split_seed(P_rec["rng_seed"], rng.SEED_SPLIT_PARTICLE)
    rng.lcg(P_rec_arr)


# ======================================================================================
# Move to event
# ======================================================================================


@njit
def move_to_event(P_arr, mcdc, data):
    settings = mcdc["settings"]

    # ==================================================================================
    # Preparation (as needed)
    # ==================================================================================
    P = P_arr[0]

    # Multigroup preparation
    #   In MG mode, particle speed is material-dependent.
    if settings["multigroup_mode"]:
        # If material is not identified yet, locate the particle
        if P["material_ID"] == -1:
            if not geometry.locate_particle(P_arr, mcdc, data):
                # Particle is lost
                P["event"] = EVENT_LOST
                return

    # ==================================================================================
    # Geometry inspection
    # ==================================================================================
    #   - Set particle top cell and material IDs (if not lost)
    #   - Set surface ID (if surface hit)
    #   - Set particle boundary event (surface or lattice crossing, or lost)
    #   - Return distance to boundary (surface or lattice)

    d_boundary = geometry.inspect_geometry(P_arr, mcdc, data)

    # Particle is lost?
    if P["event"] == EVENT_LOST:
        return

    # ==================================================================================
    # Get distances to other events
    # ==================================================================================

    # Distance to domain
    speed = physics.particle_speed(P_arr, mcdc, data)

    # Distance to time boundary
    d_time_boundary = speed * (settings["time_boundary"] - P["t"])

    # Distance to census time
    idx = mcdc["idx_census"]
    d_time_census = speed * (
        mcdc_get.settings.census_time(idx, settings, data) - P["t"]
    )

    # Distance to next collision
    d_collision = physics.collision_distance(P_arr, mcdc, data)

    # =========================================================================
    # Determine event(s)
    # =========================================================================
    # TODO: Make a function to better maintain the repeating operation

    distance = d_boundary

    # Check distance to collision
    if d_collision < distance - COINCIDENCE_TOLERANCE:
        distance = d_collision
        P["event"] = EVENT_COLLISION
        P["surface_ID"] = -1
    elif geometry.check_coincidence(d_collision, distance):
        P["event"] += EVENT_COLLISION

    # Check distance to time census
    if d_time_census < distance - COINCIDENCE_TOLERANCE:
        distance = d_time_census
        P["event"] = EVENT_TIME_CENSUS
        P["surface_ID"] = -1
    elif geometry.check_coincidence(d_time_census, distance):
        P["event"] += EVENT_TIME_CENSUS

    # Check distance to time boundary (exclusive event)
    if d_time_boundary < distance + COINCIDENCE_TOLERANCE:
        distance = d_time_boundary
        P["event"] = EVENT_TIME_BOUNDARY
        P["surface_ID"] = -1

    # =========================================================================
    # Move particle
    # =========================================================================

    # Score tracklength tallies
    if mcdc["cycle_active"]:
        # Cell tallies
        cell = mcdc["cells"][P["cell_ID"]]
        for i in range(cell["N_tally"]):
            tally_ID = int(mcdc_get.cell.tally_IDs(i, cell, data))
            tally = mcdc["cell_tallies"][tally_ID]
            tally_module.score.tracklength_tally(P_arr, distance, tally, mcdc, data)

        # Global tallies
        for i in range(mcdc["N_global_tally"]):
            tally = mcdc["global_tallies"][i]
            tally_module.score.tracklength_tally(P_arr, distance, tally, mcdc, data)

        # Mesh tallies
        for i in range(mcdc["N_mesh_tally"]):
            tally = mcdc["mesh_tallies"][i]
            tally_module.score.mesh_tally(P_arr, distance, tally, mcdc, data)

    if settings["eigenvalue_mode"]:
        tally_module.score.eigenvalue_tally(P_arr, distance, mcdc, data)

    # Move particle
    move_particle(P_arr, distance, mcdc, data)
