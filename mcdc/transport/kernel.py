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


@njit
def round(float_val):
    return float_val
    # int_val = np.float64(float_val).view(np.uint64)
    # if (int_val & 0x10) != 0:
    #    int_val += 0x10
    # int_val = int_val & ~0x0F
    # return np.uint64(int_val).view(np.float64)


# =============================================================================
# Domain Decomposition
# =============================================================================

# =============================================================================
# Domain crossing event
# =============================================================================


@toggle("domain_decomp")
def domain_crossing(P_arr, prog):
    mcdc = adapt.mcdc_global(prog)
    P = P_arr[0]
    # Domain mesh crossing
    seed = P["rng_seed"]
    max_size = mcdc["technique"]["dd_exchange_rate"]
    if mcdc["technique"]["domain_decomposition"]:
        mesh = mcdc["technique"]["dd_mesh"]
        # Determine which dimension is crossed
        ix, iy, iz, it, outside = mesh_.structured.get_indices(P_arr, mesh)

        d_idx = mcdc["dd_idx"]
        d_Nx = mcdc["technique"]["dd_mesh"]["x"].size - 1
        d_Ny = mcdc["technique"]["dd_mesh"]["y"].size - 1
        d_Nz = mcdc["technique"]["dd_mesh"]["z"].size - 1

        d_iz = int(d_idx / (d_Nx * d_Ny))
        d_iy = int((d_idx - d_Nx * d_Ny * d_iz) / d_Nx)
        d_ix = int(d_idx - d_Nx * d_Ny * d_iz - d_Nx * d_iy)

        flag = MESH_NONE
        if d_ix != ix:
            flag = MESH_X
        elif d_iy != iy:
            flag = MESH_Y
        elif d_iz != iz:
            flag = MESH_Z

        # Score on tally
        if flag == MESH_X and P["ux"] > 0:
            add_particle(P_arr, mcdc["domain_decomp"]["bank_xp"])
            if get_bank_size(mcdc["domain_decomp"]["bank_xp"]) == max_size:
                dd_initiate_particle_send(prog)
        if flag == MESH_X and P["ux"] < 0:
            add_particle(P_arr, mcdc["domain_decomp"]["bank_xn"])
            if get_bank_size(mcdc["domain_decomp"]["bank_xn"]) == max_size:
                dd_initiate_particle_send(prog)
        if flag == MESH_Y and P["uy"] > 0:
            add_particle(P_arr, mcdc["domain_decomp"]["bank_yp"])
            if get_bank_size(mcdc["domain_decomp"]["bank_yp"]) == max_size:
                dd_initiate_particle_send(prog)
        if flag == MESH_Y and P["uy"] < 0:
            add_particle(P_arr, mcdc["domain_decomp"]["bank_yn"])
            if get_bank_size(mcdc["domain_decomp"]["bank_yn"]) == max_size:
                dd_initiate_particle_send(prog)
        if flag == MESH_Z and P["uz"] > 0:
            add_particle(P_arr, mcdc["domain_decomp"]["bank_zp"])
            if get_bank_size(mcdc["domain_decomp"]["bank_zp"]) == max_size:
                dd_initiate_particle_send(prog)
        if flag == MESH_Z and P["uz"] < 0:
            add_particle(P_arr, mcdc["domain_decomp"]["bank_zn"])
            if get_bank_size(mcdc["domain_decomp"]["bank_zn"]) == max_size:
                dd_initiate_particle_send(prog)
        P["alive"] = False


# =============================================================================
# Send full domain bank
# =============================================================================


requests = []


def save_request(req_pair):
    global requests

    updated_requests = []

    status = MPI.Status()
    for req, buf in requests:
        if not req.Test(status):
            updated_requests.append((req, buf))

    updated_requests.append(req_pair)
    requests = updated_requests


def clear_requests():
    global requests
    for req, buf in requests:
        req.Free()

    requests = []


@njit
def dd_check_halt(mcdc):
    return mcdc["domain_decomp"]["work_done"]


@njit
def dd_check_in(mcdc):
    mcdc["domain_decomp"]["send_count"] = 0
    mcdc["domain_decomp"]["recv_count"] = 0
    mcdc["domain_decomp"]["send_total"] = 0
    mcdc["domain_decomp"]["rank_busy"] = True

    with objmode(rank="int64", total="int64"):
        rank = MPI.COMM_WORLD.Get_rank()
        total = MPI.COMM_WORLD.Get_size()

    if rank == 0:
        mcdc["domain_decomp"]["busy_total"] = total
    else:
        mcdc["domain_decomp"]["busy_total"] = 0


@njit
def dd_check_out(mcdc):
    with objmode():
        rank = MPI.COMM_WORLD.Get_rank()
        send_count = mcdc["domain_decomp"]["send_count"]
        recv_count = mcdc["domain_decomp"]["recv_count"]
        send_total = mcdc["domain_decomp"]["send_total"]
        busy_total = mcdc["domain_decomp"]["busy_total"]
        rank_busy = mcdc["domain_decomp"]["rank_busy"]

        if send_count != 0:
            print(
                f"Domain decomposed loop closed out with non-zero send count {send_count} in rank {rank}"
            )
            mcdc["domain_decomp"]["send_count"] = 0

        if recv_count != 0:
            print(
                f"Domain decomposed loop closed out with non-zero recv count {recv_count} in rank {rank}"
            )
            mcdc["domain_decomp"]["recv_count"] = 0

        if send_total != 0:
            print(
                f"Domain decomposed loop closed out with non-zero send total {send_total} in rank {rank}"
            )
            mcdc["domain_decomp"]["send_total"] = 0

        if busy_total != 0:
            print(
                f"Domain decomposed loop closed out with non-zero busy total {busy_total} in rank {rank}"
            )
            mcdc["domain_decomp"]["busy_total"] = 0

        if rank_busy:
            print(
                f"Domain decomposed loop closed out with rank {rank} still marked as busy"
            )
            mcdc["domain_decomp"]["rank_busy"] = 0

        clear_requests()


@njit
def dd_signal_halt(mcdc):

    with objmode():
        for rank in range(1, MPI.COMM_WORLD.Get_size()):
            dummy_buff = np.zeros((1,), dtype=np.int32)
            MPI.COMM_WORLD.Send(dummy_buff, dest=rank, tag=3)

    mcdc["domain_decomp"]["work_done"] = True


@njit
def dd_signal_block(mcdc):

    with objmode(rank="int64"):
        rank = MPI.COMM_WORLD.Get_rank()

    send_delta = (
        mcdc["domain_decomp"]["send_count"] - mcdc["domain_decomp"]["recv_count"]
    )
    if rank == 0:
        mcdc["domain_decomp"]["send_total"] += send_delta
        mcdc["domain_decomp"]["busy_total"] -= 1
    else:
        with objmode():
            buff = np.zeros((1,), dtype=type_.dd_turnstile_event)
            buff[0]["busy_delta"] = -1
            buff[0]["send_delta"] = send_delta
            req = MPI.COMM_WORLD.Isend(
                [buff, type_.dd_turnstile_event_mpi], dest=0, tag=2
            )
            save_request((req, buff))

    mcdc["domain_decomp"]["send_count"] = 0
    mcdc["domain_decomp"]["recv_count"] = 0

    if (
        (rank == 0)
        and (mcdc["domain_decomp"]["busy_total"] == 0)
        and (mcdc["domain_decomp"]["send_total"] == 0)
    ):
        dd_signal_halt(mcdc)


@njit
def dd_signal_unblock(mcdc):

    with objmode(rank="int64"):
        rank = MPI.COMM_WORLD.Get_rank()

    send_delta = (
        mcdc["domain_decomp"]["send_count"] - mcdc["domain_decomp"]["recv_count"]
    )

    if rank == 0:
        mcdc["domain_decomp"]["send_total"] += send_delta
        mcdc["domain_decomp"]["busy_total"] += 1
        if (mcdc["domain_decomp"]["busy_total"] == 0) and (
            mcdc["domain_decomp"]["send_total"] == 0
        ):
            dd_signal_halt(mcdc)
    else:
        with objmode():
            buff = np.zeros((1,), dtype=type_.dd_turnstile_event)
            buff[0]["busy_delta"] = 1
            buff[0]["send_delta"] = send_delta
            req = MPI.COMM_WORLD.Isend(
                [buff, type_.dd_turnstile_event_mpi], dest=0, tag=2
            )
            save_request((req, buff))
    mcdc["domain_decomp"]["send_count"] = 0
    mcdc["domain_decomp"]["recv_count"] = 0


@njit
def dd_distribute_bank(mcdc, bank, dest_list):

    with objmode(send_delta="int64"):
        dest_count = len(dest_list)
        send_delta = 0

        for i, dest in enumerate(dest_list):
            size = get_bank_size(bank)
            ratio = int(size / dest_count)
            start = ratio * i
            end = start + ratio
            if i == dest_count - 1:
                end = size
            sub_bank = np.array(bank["particles"][start:end])
            if sub_bank.shape[0] > 0:
                req = MPI.COMM_WORLD.Isend(
                    [sub_bank, type_.particle_data_mpi], dest=dest, tag=1
                )
                save_request((req, sub_bank))
                send_delta += end - start

    mcdc["domain_decomp"]["send_count"] += send_delta
    set_bank_size(bank, 0)


@for_gpu()
def dd_initiate_particle_send(prog):
    adapt.halt_early(prog)


@for_cpu()
def dd_initiate_particle_send(prog):
    dd_particle_send(prog)


@njit
def dd_particle_send(prog):
    mcdc = adapt.mcdc_global(prog)
    dd_distribute_bank(
        mcdc, mcdc["domain_decomp"]["bank_xp"], mcdc["technique"]["dd_xp_neigh"]
    )
    dd_distribute_bank(
        mcdc, mcdc["domain_decomp"]["bank_xn"], mcdc["technique"]["dd_xn_neigh"]
    )
    dd_distribute_bank(
        mcdc, mcdc["domain_decomp"]["bank_yp"], mcdc["technique"]["dd_yp_neigh"]
    )
    dd_distribute_bank(
        mcdc, mcdc["domain_decomp"]["bank_yn"], mcdc["technique"]["dd_yn_neigh"]
    )
    dd_distribute_bank(
        mcdc, mcdc["domain_decomp"]["bank_zp"], mcdc["technique"]["dd_zp_neigh"]
    )
    dd_distribute_bank(
        mcdc, mcdc["domain_decomp"]["bank_zn"], mcdc["technique"]["dd_zn_neigh"]
    )


# =============================================================================
# Receive particles and clear banks
# =============================================================================


@njit
def dd_get_recv_tag():

    with objmode(tag="int64"):
        status = MPI.Status()
        MPI.COMM_WORLD.Probe(status=status)
        tag = status.Get_tag()

    return tag


@njit
def dd_recv_particles(mcdc):

    buff = np.zeros(
        mcdc["domain_decomp"]["bank_zp"]["particles"].shape[0],
        dtype=type_.particle_data,
    )

    with objmode(size="int64"):
        status = MPI.Status()
        MPI.COMM_WORLD.Recv([buff, type_.particle_data_mpi], status=status)
        size = status.Get_count(type_.particle_data_mpi)
        rank = MPI.COMM_WORLD.Get_rank()

    mcdc["domain_decomp"]["recv_count"] += size

    # Set source bank from buffer
    for i in range(size):
        add_particle(buff[i : i + 1], mcdc["bank_active"])

    if (
        mcdc["domain_decomp"]["recv_count"] > 0
        and not mcdc["domain_decomp"]["rank_busy"]
    ):
        dd_signal_unblock(mcdc)
        mcdc["domain_decomp"]["rank_busy"] = True


@njit
def dd_recv_turnstile(mcdc):

    with objmode(busy_delta="int64", send_delta="int64"):
        event_buff = np.zeros((1,), dtype=type_.dd_turnstile_event)
        MPI.COMM_WORLD.Recv([event_buff, type_.dd_turnstile_event_mpi])
        busy_delta = event_buff[0]["busy_delta"]
        send_delta = event_buff[0]["send_delta"]
        rank = MPI.COMM_WORLD.Get_rank()
        busy_total = mcdc["domain_decomp"]["busy_total"]
        send_total = mcdc["domain_decomp"]["send_total"]

    mcdc["domain_decomp"]["busy_total"] += busy_delta
    mcdc["domain_decomp"]["send_total"] += send_delta

    if (mcdc["domain_decomp"]["busy_total"] == 0) and (
        mcdc["domain_decomp"]["send_total"] == 0
    ):
        dd_signal_halt(mcdc)


@njit
def dd_recv_halt(mcdc):

    with objmode():
        dummy_buff = np.zeros((1,), dtype=np.int32)
        MPI.COMM_WORLD.Recv(dummy_buff)
        work_done = 1
        rank = MPI.COMM_WORLD.Get_rank()

    mcdc["domain_decomp"]["work_done"] = True


@njit
def dd_recv(mcdc):

    if mcdc["domain_decomp"]["rank_busy"]:
        dd_signal_block(mcdc)
        mcdc["domain_decomp"]["rank_busy"] = False

    if not mcdc["domain_decomp"]["work_done"]:
        tag = dd_get_recv_tag()

        if tag == 1:
            dd_recv_particles(mcdc)
        elif tag == 2:
            dd_recv_turnstile(mcdc)
        elif tag == 3:
            dd_recv_halt(mcdc)


# =============================================================================
# Particle in domain
# =============================================================================


# Check if particle is in domain
@njit
def particle_in_domain(P_arr, mcdc):
    P = P_arr[0]
    d_idx = mcdc["dd_idx"]
    d_Nx = mcdc["technique"]["dd_mesh"]["x"].size - 1
    d_Ny = mcdc["technique"]["dd_mesh"]["y"].size - 1
    d_Nz = mcdc["technique"]["dd_mesh"]["z"].size - 1

    d_iz = int(d_idx / (d_Nx * d_Ny))
    d_iy = int((d_idx - d_Nx * d_Ny * d_iz) / d_Nx)
    d_ix = int(d_idx - d_Nx * d_Ny * d_iz - d_Nx * d_iy)

    mesh = mcdc["technique"]["dd_mesh"]
    x_cell, y_cell, z_cell, t_cell, outside = mesh_.structured.get_indices(P_arr, mesh)

    if d_ix == x_cell:
        if d_iy == y_cell:
            if d_iz == z_cell:
                return True
    return False


# =============================================================================
# Source in domain
# =============================================================================


# Check for source in domain
@njit
def source_in_domain(source, domain_mesh, d_idx):
    d_Nx = domain_mesh["x"].size - 1
    d_Ny = domain_mesh["y"].size - 1
    d_Nz = domain_mesh["z"].size - 1

    d_iz = int(d_idx / (d_Nx * d_Ny))
    d_iy = int((d_idx - d_Nx * d_Ny * d_iz) / d_Nx)
    d_ix = int(d_idx - d_Nx * d_Ny * d_iz - d_Nx * d_iy)

    d_x = [domain_mesh["x"][d_ix], domain_mesh["x"][d_ix + 1]]
    d_y = [domain_mesh["y"][d_iy], domain_mesh["y"][d_iy + 1]]
    d_z = [domain_mesh["z"][d_iz], domain_mesh["z"][d_iz + 1]]

    if (
        d_x[0] <= source["box_x"][0] <= d_x[1]
        or d_x[0] <= source["box_x"][1] <= d_x[1]
        or (source["box_x"][0] < d_x[0] and source["box_x"][1] > d_x[1])
    ):
        if (
            d_y[0] <= source["box_y"][0] <= d_y[1]
            or d_y[0] <= source["box_y"][1] <= d_y[1]
            or (source["box_y"][0] < d_y[0] and source["box_y"][1] > d_y[1])
        ):
            if (
                d_z[0] <= source["box_z"][0] <= d_z[1]
                or d_z[0] <= source["box_z"][1] <= d_z[1]
                or (source["box_z"][0] < d_z[0] and source["box_z"][1] > d_z[1])
            ):
                return True
            else:
                return False
        else:
            return False
    else:
        return False


# =============================================================================
# Compute domain load
# =============================================================================


@njit
def domain_work(mcdc, domain, N):
    domain_mesh = mcdc["technique"]["dd_mesh"]

    d_Nx = domain_mesh["x"].size - 1
    d_Ny = domain_mesh["y"].size - 1
    d_Nz = domain_mesh["z"].size - 1
    work_start = 0
    for d_idx in range(domain):
        d_iz = int(d_idx / (d_Nx * d_Ny))
        d_iy = int((d_idx - d_Nx * d_Ny * d_iz) / d_Nx)
        d_ix = int(d_idx - d_Nx * d_Ny * d_iz - d_Nx * d_iy)

        d_x = [domain_mesh["x"][d_ix], domain_mesh["x"][d_ix + 1]]
        d_y = [domain_mesh["y"][d_iy], domain_mesh["y"][d_iy + 1]]
        d_z = [domain_mesh["z"][d_iz], domain_mesh["z"][d_iz + 1]]
        # Compute volumes of sources and numbers of particles

        Psum = 0

        Nm = 0
        num_source = 0
        for source in mcdc["sources"]:
            Psum += source["prob"]
            num_source += 1
        Vi = np.zeros(num_source)
        Vim = np.zeros(num_source)
        Ni = np.zeros(num_source)
        i = 0
        for source in mcdc["sources"]:
            Ni[i] = N * source["prob"] / Psum
            Vi[i] = 1
            Vim[i] = 1
            if source["box"] == True:
                xV = source["box_x"][1] - source["box_x"][0]
                if xV != 0:
                    Vi[i] *= xV
                    Vim[i] *= min(source["box_x"][1], d_x[1]) - max(
                        source["box_x"][0], d_x[0]
                    )
                yV = source["box_y"][1] - source["box_y"][0]
                if yV != 0:
                    Vi[i] *= yV
                    Vim[i] *= min(source["box_y"][1], d_y[1]) - max(
                        source["box_y"][0], d_y[0]
                    )
                zV = source["box_z"][1] - source["box_z"][0]
                if zV != 0:
                    Vi[i] *= zV
                    Vim[i] *= min(source["box_z"][1], d_z[1]) - max(
                        source["box_z"][0], d_z[0]
                    )
            if not source_in_domain(source, domain_mesh, d_idx):
                Vim[i] = 0
            i += 1
        for source in range(num_source):
            Nm += Ni[source] * Vim[source] / Vi[source]
        work_start += Nm
    d_idx = domain
    d_iz = int(mcdc["dd_idx"] / (d_Nx * d_Ny))
    d_iy = int((mcdc["dd_idx"] - d_Nx * d_Ny * d_iz) / d_Nx)
    d_ix = int(mcdc["dd_idx"] - d_Nx * d_Ny * d_iz - d_Nx * d_iy)

    d_x = [domain_mesh["x"][d_ix], domain_mesh["x"][d_ix + 1]]
    d_y = [domain_mesh["y"][d_iy], domain_mesh["y"][d_iy + 1]]
    d_z = [domain_mesh["z"][d_iz], domain_mesh["z"][d_iz + 1]]
    # Compute volumes of sources and numbers of particles
    num_source = len(mcdc["sources"])
    Vi = np.zeros(num_source)
    Vim = np.zeros(num_source)
    Ni = np.zeros(num_source)
    Psum = 0

    Nm = 0
    for source in mcdc["sources"]:
        Psum += source["prob"]
    i = 0
    for source in mcdc["sources"]:
        Ni[i] = N * source["prob"] / Psum
        Vi[i] = 1
        Vim[i] = 1

        if source["box"] == True:
            xV = source["box_x"][1] - source["box_x"][0]
            if xV != 0:
                Vi[i] *= xV
                Vim[i] *= min(source["box_x"][1], d_x[1]) - max(
                    source["box_x"][0], d_x[0]
                )
            yV = source["box_y"][1] - source["box_y"][0]
            if yV != 0:
                Vi[i] *= yV
                Vim[i] *= min(source["box_y"][1], d_y[1]) - max(
                    source["box_y"][0], d_y[0]
                )
            zV = source["box_z"][1] - source["box_z"][0]
            if zV != 0:
                Vi[i] *= zV
                Vim[i] *= min(source["box_z"][1], d_z[1]) - max(
                    source["box_z"][0], d_z[0]
                )
        i += 1
    for source in range(num_source):
        Nm += Ni[source] * Vim[source] / Vi[source]
    Nm /= mcdc["technique"]["dd_work_ratio"][domain]
    rank = mcdc["mpi_rank"]
    if mcdc["technique"]["dd_work_ratio"][domain] > 1:
        work_start += Nm * (rank - np.sum(mcdc["technique"]["dd_work_ratio"][0:d_idx]))
    total_v = 0
    for source in range(len(mcdc["sources"])):
        total_v += Vim[source]
    i = 0
    for source in mcdc["sources"]:
        if total_v != 0:
            source["prob"] *= 2 * Vim[i] / total_v
        i += 1
    return (int(Nm), int(work_start))


# =============================================================================
# Random sampling
# =============================================================================


@njit
def sample_piecewise_linear(cdf, P_arr):
    P = P_arr[0]
    xi = rng(P_arr)

    # Get bin
    idx = binary_search(xi, cdf[1])

    # Linear interpolation
    x1 = cdf[1, idx]
    x2 = cdf[1, idx + 1]
    y1 = cdf[0, idx]
    y2 = cdf[0, idx + 1]
    return y1 + (xi - x1) * (y2 - y1) / (x2 - x1)


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
    P_arr = adapt.local_array(1, type_.particle_data)
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
def manage_particle_banks(seed, mcdc):
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
    if mcdc["population_control"]['active']:
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
    for P in bank["particles"]:
        P["w"] *= norm / W


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

    distribute_work(N, mcdc)

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


@njit
def distribute_work(N, mcdc):
    size = mcdc["mpi_size"]
    rank = mcdc["mpi_rank"]

    # Total # of work
    work_size_total = N

    # Evenly distribute work
    work_size = math.floor(N / size)

    # Starting index (based on even distribution)
    work_start = work_size * rank

    # Count reminder
    rem = N % size

    # Assign reminder and update starting index
    if rank < rem:
        work_size += 1
        work_start += rank
    else:
        work_start += rem

    mcdc["mpi_work_start"] = work_start
    mcdc["mpi_work_size"] = work_size
    mcdc["mpi_work_size_total"] = work_size_total


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
    P_new["iqmc"]["w"] = P["iqmc"]["w"]


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


# =============================================================================
# Tally operations
# =============================================================================

@njit
def dd_reduce(data, mcdc):

    # find number of subdomains
    d_Nx = mcdc["technique"]["dd_mesh"]["x"].size - 1
    d_Ny = mcdc["technique"]["dd_mesh"]["y"].size - 1
    d_Nz = mcdc["technique"]["dd_mesh"]["z"].size - 1

    with objmode():
        # assign processors to their subdomain group
        i = 0
        for n in range(d_Nx * d_Ny * d_Nz):
            dd_group = []
            for r in range(int(mcdc["technique"]["dd_work_ratio"][n])):
                dd_group.append(i)
                i += 1
            # create MPI Comm group out of subdomain processors
            dd_group = MPI.COMM_WORLD.group.Incl(dd_group)
            dd_comm = MPI.COMM_WORLD.Create(dd_group)
            # MPI Reduce on subdomain processors
            buff = np.zeros_like(data[TALLY_SCORE])
            if MPI.COMM_NULL != dd_comm:
                dd_comm.Reduce(data[TALLY_SCORE], buff, MPI.SUM, 0)
            if mcdc["dd_idx"] == n:
                data[TALLY_SCORE][:] = buff
            # free comm group
            dd_group.Free()
            if MPI.COMM_NULL != dd_comm:
                dd_comm.Free()


@njit
def census_based_tally_output(data, mcdc):
    idx_batch = mcdc["idx_batch"]
    idx_census = mcdc["idx_census"]
    N_bin = data.shape[1]

    for i in range(N_bin):
        # Store score and square of score
        score = data[TALLY_SCORE, i]
        data[TALLY_SUM, i] = score
        data[TALLY_SUM_SQ, i] = score * score

        # Reset score bin
        data[TALLY_SCORE, i] = 0.0

    for ID, tally in enumerate(mcdc["mesh_tallies"]):
        mesh = tally["filter"]

        # Get grid
        Nx = mesh["Nx"]
        Ny = mesh["Ny"]
        Nz = mesh["Nz"]
        Nt = mesh["Nt"]
        Nmu = mesh["Nmu"]
        N_azi = mesh["N_azi"]
        Ng = mesh["Ng"]
        #
        grid_x = mesh["x"][: Nx + 1]
        grid_y = mesh["y"][: Ny + 1]
        grid_z = mesh["z"][: Nz + 1]
        grid_t = mesh["t"][: Nt + 1]
        grid_mu = mesh["mu"][: Nmu + 1]
        grid_azi = mesh["azi"][: N_azi + 1]
        grid_g = mesh["g"][: Ng + 1]
        #'''
        with objmode():
            if ID == 0:
                f = h5py.File(
                    mcdc["settings"]["output_name"]
                    + "-batch_%i-census_%i.h5" % (idx_batch, idx_census),
                    "w",
                )
            else:
                f = h5py.File(
                    mcdc["settings"]["output_name"]
                    + "-batch_%i-census_%i.h5" % (idx_batch, idx_census),
                    "a",
                )
            # Save to dataset
            f.create_dataset("tallies/mesh_tally_%i/grid/x" % ID, data=grid_x)
            f.create_dataset("tallies/mesh_tally_%i/grid/y" % ID, data=grid_y)
            f.create_dataset("tallies/mesh_tally_%i/grid/z" % ID, data=grid_z)
            f.create_dataset("tallies/mesh_tally_%i/grid/t" % ID, data=grid_t)
            f.create_dataset("tallies/mesh_tally_%i/grid/mu" % ID, data=grid_mu)
            f.create_dataset("tallies/mesh_tally_%i/grid/azi" % ID, data=grid_azi)
            f.create_dataset("tallies/mesh_tally_%i/grid/g" % ID, data=grid_g)

            # Set tally shape
            N_score = tally["N_score"]
            if not mcdc["technique"]["uq"]:
                shape = (3, Nmu, N_azi, Ng, Nt, Nx, Ny, Nz, N_score)
            else:
                shape = (5, Nmu, N_azi, Ng, Nt, Nx, Ny, Nz, N_score)

            # Reshape tally
            N_bin = tally["N_bin"]
            start = tally["stride"]["tally"]
            tally_bin = data[:, start : start + N_bin]
            tally_bin = tally_bin.reshape(shape)

            # Roll tally so that score is in the front
            tally_bin = np.rollaxis(tally_bin, 8, 0)

            # Iterate over scores
            for i in range(N_score):
                score_type = tally["scores"][i]
                score_tally_bin = np.squeeze(tally_bin[i])
                score_name = ""
                if score_type == SCORE_FLUX:
                    score_name = "flux"
                elif score_type == SCORE_DENSITY:
                    score_name = "density"
                elif score_type == SCORE_TOTAL:
                    score_name = "total"
                elif score_type == SCORE_FISSION:
                    score_name = "fission"
                group_name = "tallies/mesh_tally_%i/%s/" % (ID, score_name)

                tally_sum = score_tally_bin[TALLY_SUM]
                tally_sum_sq = score_tally_bin[TALLY_SUM_SQ]

                f.create_dataset(group_name + "score", data=tally_sum)
                f.create_dataset(group_name + "score_sq", data=tally_sum_sq)
                if mcdc["technique"]["uq"]:
                    mc_var = score_tally_bin[TALLY_UQ_BATCH_VAR]
                    tot_var = score_tally_bin[TALLY_UQ_BATCH]
                    uq_var = tot_var - mc_var
                    f.create_dataset(group_name + "uq_var", data=uq_var)
            f.close()


@njit
def dd_closeout(data, mcdc):
    # find number of subdomains
    d_Nx = mcdc["technique"]["dd_mesh"]["x"].size - 1
    d_Ny = mcdc["technique"]["dd_mesh"]["y"].size - 1
    d_Nz = mcdc["technique"]["dd_mesh"]["z"].size - 1

    with objmode():
        # assign processors to their subdomain group
        i = 0
        for n in range(d_Nx * d_Ny * d_Nz):
            dd_ranks = []
            for r in range(int(mcdc["technique"]["dd_work_ratio"][n])):
                dd_ranks.append(i)
                i += 1
            # create MPI Comm group out of subdomain processors
            dd_group = MPI.COMM_WORLD.group.Incl(dd_ranks)
            dd_comm = MPI.COMM_WORLD.Create(dd_group)
            # MPI Reduce on subdomain processors
            buff = np.zeros_like(data[TALLY_SUM])
            buff_sq = np.zeros_like(data[TALLY_SUM_SQ])
            if MPI.COMM_NULL != dd_comm:
                dd_comm.Reduce(data[TALLY_SUM], buff, MPI.SUM, 0)
                dd_comm.Reduce(data[TALLY_SUM_SQ], buff_sq, MPI.SUM, 0)
            if mcdc["dd_idx"] == n:
                data[TALLY_SUM] = buff
                data[TALLY_SUM_SQ] = buff_sq

            # free comm group
            dd_group.Free()
            if MPI.COMM_NULL != dd_comm:
                dd_comm.Free()


# =============================================================================
# Eigenvalue tally operations
# =============================================================================


@njit
def eigenvalue_tally(P_arr, distance, mcdc, data):
    P = P_arr[0]
    material = mcdc["materials"][P["material_ID"]]
    flux = distance * P["w"]

    # Get nu-fission
    nuSigmaF = physics.neutron_production_xs(
        REACTION_NEUTRON_FISSION, material, P_arr, mcdc, data
    )

    adapt.global_add(mcdc["eigenvalue_tally_nuSigmaF"], 0, round(flux * nuSigmaF))
    return

    # TODO
    # Fission production (needed even during inactive cycle)
    # mcdc["eigenvalue_tally_nuSigmaF"][0] += flux * nuSigmaF

    if mcdc["cycle_active"]:
        # Neutron density
        v = physics.particle_speed(P_arr, material, data)
        n_density = flux / v
        # mcdc["eigenvalue_tally_n"][0] += n_density
        adapt.global_add(mcdc["eigenvalue_tally_n"], 0, round(n_density))
        # Maximum neutron density
        if mcdc["n_max"] < n_density:
            mcdc["n_max"] = n_density

        # Precursor density
        J = material["J"]
        SigmaF = physics.macro_xs(REACTION_NEUTRON_FISSION, material, P_arr, mcdc, data)
        # Get the decay-wighted multiplicity
        total = 0.0
        if mcdc["settings"]["multigroup_mode"]:
            g = P["g"]
            for j in range(J):
                nu_d = mcdc_get.material.mgxs_nu_d(g, j, material, data)
                decay = mcdc_get.material.mgxs_decay_rate(j, material, data)
                total += nu_d / decay
        else:
            E = P["E"]
            for i in range(material["N_nuclide"]):
                ID_nuclide = material["nuclide_IDs"][i]
                nuclide = mcdc["nuclides"][ID_nuclide]
                if not nuclide["fissionable"]:
                    continue
                for j in range(J):
                    # HAZARD
                    nu_d = get_nu_group(NU_FISSION_DELAYED, nuclide, E, j)
                    decay = nuclide["ce_decay"][j]
                    total += nu_d / decay
        C_density = flux * total * SigmaF / mcdc["k_eff"]
        # mcdc["eigenvalue_tally_C"][0] += C_density
        adapt.global_add(mcdc["eigenvalue_tally_C"], 0, round(C_density))
        # Maximum precursor density
        if mcdc["C_max"] < C_density:
            mcdc["C_max"] = C_density


@njit
def eigenvalue_tally_closeout_history(mcdc):
    N_particle = mcdc["settings"]["N_particle"]

    idx_cycle = mcdc["idx_cycle"]

    # MPI Allreduce
    buff_nuSigmaF = np.zeros(1, np.float64)
    buff_n = np.zeros(1, np.float64)
    buff_nmax = np.zeros(1, np.float64)
    buff_C = np.zeros(1, np.float64)
    buff_Cmax = np.zeros(1, np.float64)
    with objmode():
        MPI.COMM_WORLD.Allreduce(
            np.array(mcdc["eigenvalue_tally_nuSigmaF"]), buff_nuSigmaF, MPI.SUM
        )
        if mcdc["cycle_active"]:
            MPI.COMM_WORLD.Allreduce(
                np.array(mcdc["eigenvalue_tally_n"]), buff_n, MPI.SUM
            )
            MPI.COMM_WORLD.Allreduce(np.array([mcdc["n_max"]]), buff_nmax, MPI.MAX)
            MPI.COMM_WORLD.Allreduce(
                np.array(mcdc["eigenvalue_tally_C"]), buff_C, MPI.SUM
            )
            MPI.COMM_WORLD.Allreduce(np.array([mcdc["C_max"]]), buff_Cmax, MPI.MAX)

    # Update and store k_eff
    mcdc["k_eff"] = buff_nuSigmaF[0] / N_particle
    mcdc["k_cycle"][idx_cycle] = mcdc["k_eff"]

    # Normalize other eigenvalue/global tallies
    tally_n = buff_n[0] / N_particle
    tally_C = buff_C[0] / N_particle

    # Maximum densities
    mcdc["n_max"] = buff_nmax[0]
    mcdc["C_max"] = buff_Cmax[0]

    # Accumulate running average
    if mcdc["cycle_active"]:
        mcdc["k_avg"] += mcdc["k_eff"]
        mcdc["k_sdv"] += mcdc["k_eff"] * mcdc["k_eff"]
        mcdc["n_avg"] += tally_n
        mcdc["n_sdv"] += tally_n * tally_n
        mcdc["C_avg"] += tally_C
        mcdc["C_sdv"] += tally_C * tally_C

        N = 1 + mcdc["idx_cycle"] - mcdc["settings"]["N_inactive"]
        mcdc["k_avg_running"] = mcdc["k_avg"] / N
        if N == 1:
            mcdc["k_sdv_running"] = 0.0
        else:
            mcdc["k_sdv_running"] = math.sqrt(
                (mcdc["k_sdv"] / N - mcdc["k_avg_running"] ** 2) / (N - 1)
            )

    # Reset accumulators
    mcdc["eigenvalue_tally_nuSigmaF"][0] = 0.0
    mcdc["eigenvalue_tally_n"][0] = 0.0
    mcdc["eigenvalue_tally_C"][0] = 0.0

    # =====================================================================
    # Gyration radius
    # =====================================================================

    if mcdc["settings"]["use_gyration_radius"]:
        # Center of mass
        N_local = get_bank_size(mcdc["bank_census"])
        total_local = np.zeros(4, np.float64)  # [x,y,z,W]
        total = np.zeros(4, np.float64)
        for i in range(N_local):
            P = mcdc["bank_census"]["particles"][i]
            total_local[0] += P["x"] * P["w"]
            total_local[1] += P["y"] * P["w"]
            total_local[2] += P["z"] * P["w"]
            total_local[3] += P["w"]
        # MPI Allreduce
        with objmode():
            MPI.COMM_WORLD.Allreduce(total_local, total, MPI.SUM)
        # COM
        W = total[3]
        com_x = total[0] / W
        com_y = total[1] / W
        com_z = total[2] / W

        # Distance RMS
        rms_local = np.zeros(1, np.float64)
        rms = np.zeros(1, np.float64)
        gr_type = mcdc["settings"]["gyration_radius_type"]
        if gr_type == GYRATION_RADIUS_ALL:
            for i in range(N_local):
                P = mcdc["bank_census"]["particles"][i]
                rms_local[0] += (
                    (P["x"] - com_x) ** 2
                    + (P["y"] - com_y) ** 2
                    + (P["z"] - com_z) ** 2
                ) * P["w"]
        elif gr_type == GYRATION_RADIUS_INFINITE_X:
            for i in range(N_local):
                P = mcdc["bank_census"]["particles"][i]
                rms_local[0] += ((P["y"] - com_y) ** 2 + (P["z"] - com_z) ** 2) * P["w"]
        elif gr_type == GYRATION_RADIUS_INFINITE_Y:
            for i in range(N_local):
                P = mcdc["bank_census"]["particles"][i]
                rms_local[0] += ((P["x"] - com_x) ** 2 + (P["z"] - com_z) ** 2) * P["w"]
        elif gr_type == GYRATION_RADIUS_INFINITE_Z:
            for i in range(N_local):
                P = mcdc["bank_census"]["particles"][i]
                rms_local[0] += ((P["x"] - com_x) ** 2 + (P["y"] - com_y) ** 2) * P["w"]
        elif gr_type == GYRATION_RADIUS_ONLY_X:
            for i in range(N_local):
                P = mcdc["bank_census"]["particles"][i]
                rms_local[0] += ((P["x"] - com_x) ** 2) * P["w"]
        elif gr_type == GYRATION_RADIUS_ONLY_Y:
            for i in range(N_local):
                P = mcdc["bank_census"]["particles"][i]
                rms_local[0] += ((P["y"] - com_y) ** 2) * P["w"]
        elif gr_type == GYRATION_RADIUS_ONLY_Z:
            for i in range(N_local):
                P = mcdc["bank_census"]["particles"][i]
                rms_local[0] += ((P["z"] - com_z) ** 2) * P["w"]

        # MPI Allreduce
        with objmode():
            MPI.COMM_WORLD.Allreduce(rms_local, rms, MPI.SUM)
        rms = math.sqrt(rms[0] / W)

        # Gyration radius
        mcdc["gyration_radius"][idx_cycle] = rms


@njit
def eigenvalue_tally_closeout(mcdc):
    N = mcdc["settings"]["N_active"]
    mcdc["n_avg"] /= N
    mcdc["C_avg"] /= N
    if N > 1:
        mcdc["n_sdv"] = math.sqrt((mcdc["n_sdv"] / N - mcdc["n_avg"] ** 2) / (N - 1))
        mcdc["C_sdv"] = math.sqrt((mcdc["C_sdv"] / N - mcdc["C_avg"] ** 2) / (N - 1))
    else:
        mcdc["n_sdv"] = 0.0
        mcdc["C_sdv"] = 0.0


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
    d_domain = INF

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

    # Check distance to domain
    if d_domain < distance - COINCIDENCE_TOLERANCE:
        distance = d_domain
        P["event"] = EVENT_DOMAIN_CROSSING
        P["surface_ID"] = -1
    elif geometry.check_coincidence(d_domain, distance):
        P["event"] += EVENT_DOMAIN_CROSSING

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
            tally_module.score.cell_tally(P_arr, distance, tally, mcdc, data)
        
        # Global tallies
        for i in range(mcdc['N_global_tally']):
            tally = mcdc["global_tallies"][i]
            tally_module.score.cell_tally(P_arr, distance, tally, mcdc, data)

        # Mesh tallies
        for tally in mcdc["mesh_tallies"]:
            tally_module.score.mesh_tally(P_arr, distance, tally, mcdc, data)

    if settings["eigenvalue_mode"]:
        eigenvalue_tally(P_arr, distance, mcdc, data)

    # Move particle
    move_particle(P_arr, distance, mcdc, data)


# =============================================================================
# Weight window
# =============================================================================


@njit
def weight_window(P_arr, prog):
    P = P_arr[0]
    mcdc = adapt.mcdc_global(prog)

    # Get indices
    ix, iy, iz, it, outside = mesh_.structured.get_indices(
        P_arr, mcdc["technique"]["ww"]["mesh"]
    )

    # Target weight
    w_target = mcdc["technique"]["ww"]["center"][it, ix, iy, iz]

    # Population control factor
    w_target *= mcdc["technique"]["pc_factor"]

    # Surviving probability
    p = P["w"] / w_target

    # Window width
    width = mcdc["technique"]["ww"]["width"]

    P_new_arr = adapt.local_array(1, type_.particle_data)

    # If above target
    if p > width:
        # Set target weight
        P["w"] = w_target

        # Splitting (keep the original particle)
        n_split = math.floor(p)
        for i in range(n_split - 1):
            split_as_data(P_new_arr, P_arr)
            adapt.add_active(P_new_arr, prog)

        # Russian roulette
        p -= n_split
        xi = rng(P_arr)
        if xi <= p:
            split_as_data(P_new_arr, P_arr)
            adapt.add_active(P_new_arr, prog)

    # Below target
    elif p < 1.0 / width:
        # Russian roulette
        xi = rng(P_arr)
        if xi > p:
            P["alive"] = False
        else:
            P["w"] = w_target


@njit
def update_weight_windows(data, mcdc):
    idx_batch = mcdc["idx_batch"]
    idx_census = mcdc["idx_census"]
    epsilon = mcdc["technique"]["ww"]["epsilon"]
    # accessing most recent tally dump
    with objmode():
        f = h5py.File(
            mcdc["settings"]["output_name"]
            + "-batch_%i-census_%i.h5" % (idx_batch, idx_census),
            "r",
        )
        tallies = f["tallies/mesh_tally_" + str(mcdc["technique"]["ww"]["tally_idx"])]
        if mcdc["settings"]["census_tally_frequency"] > 1:
            old_flux = tallies["flux"]["score"][-1]
        else:
            old_flux = tallies["flux"]["score"]
        Nx = mcdc["technique"]["ww"]["mesh"]["Nx"]
        Ny = mcdc["technique"]["ww"]["mesh"]["Ny"]
        Nz = mcdc["technique"]["ww"]["mesh"]["Nz"]
        Nt = mcdc["technique"]["ww"]["mesh"]["Nt"]

        ax_expand = []
        if Nx == 1:
            ax_expand.append(0)
        if Ny == 1:
            ax_expand.append(1)
        if Nz == 1:
            ax_expand.append(2)
        for ax in ax_expand:
            old_flux = np.expand_dims(old_flux, axis=ax)
        center = old_flux

        if epsilon[WW_WOLLABER] > 0:
            w_min = epsilon[WW_WOLLABER + 1]
            center = (center) * (
                1
                + (1 / epsilon[WW_WOLLABER] - 1)
                * np.exp(-(center - w_min) / epsilon[WW_WOLLABER])
            )
        if epsilon[WW_MIN] > 0:
            center = center * (1 - epsilon[WW_MIN]) + epsilon[WW_MIN]
            center[center <= 0] = epsilon[WW_MIN]
        center /= np.max(center)
        mcdc["technique"]["ww"]["center"][idx_census + 1] = center


# =============================================================================
# Weight Roulette
# =============================================================================


@njit
def weight_roulette(P_arr, mcdc):
    P = P_arr[0]
    w_survive = mcdc["technique"]["wr_survive"]
    prob_survive = P["w"] / w_survive
    if rng(P_arr) <= prob_survive:
        P["w"] = w_survive
        if mcdc["technique"]["iQMC"]:
            P["iqmc"]["w"][:] = w_survive
    else:
        P["alive"] = False
