from mcdc.code_factory.code_factory import plural_to_singular, singular_to_plural
from mpi4py import MPI

####

import mcdc.code_factory as code_factory
import mcdc.config as config
import mcdc.transport.physics as physics
import mcdc.mcdc_get as mcdc_get

from mcdc.object_.material import MaterialMG
from mcdc.object_.simulation import simulation
from mcdc.print_ import (
    print_banner,
    print_configuration,
    print_eigenvalue_header,
    print_error,
    print_msg,
    print_runtime,
    print_structure,
)
from mcdc.transport.simulation import (
    fixed_source_simulation,
    eigenvalue_simulation,
    build_gpu_progs,
    teardown_gpu,
    setup_gpu,
)

########################################################################################

import importlib.metadata
import matplotlib.pyplot as plt
import numba as nb
from matplotlib import colors as mpl_colors
import scipy.fft as spfft
from scipy.stats.qmc import Halton
import cvxpy as cp

import h5py
import numpy as np


import mcdc.transport.kernel as kernel
import mcdc.togo.type_ as type_

import mcdc.code_factory.adapt as adapt
from mcdc.constant import *
import mcdc.transport.geometry as geometry

# Get input_deck
import mcdc.togo.global_ as mcdc_

input_deck = mcdc_.input_deck


# ======================================================================================
# Run
# ======================================================================================


def run():
    # Timer: total
    time_total_start = MPI.Wtime()

    settings = simulation.settings
    master = MPI.COMM_WORLD.Get_rank() == 0

    # ==================================================================================
    # Override settings with command-line arguments
    # ==================================================================================

    if config.args.N_particle is not None:
        settings.N_particle = config.args.N_particle
    if config.args.output is not None:
        settings.output_name = config.args.output
    if config.args.progress_bar is not None:
        settings.use_progress_bar = config.args.progress_bar

    # ==================================================================================
    # Preparation
    # ==================================================================================

    # Timer: preparation
    time_prep_start = MPI.Wtime()

    mcdc_arr, data = prepare()
    mcdc = mcdc_arr[0]

    # Print headers
    if master:
        print_banner()
        print_configuration()
        print(" Now running TNT...")
        if settings.eigenvalue_mode:
            print_eigenvalue_header(mcdc)

    # Timer: preparation
    time_prep_end = MPI.Wtime()

    # ==================================================================================
    # Running the simulation
    # ==================================================================================

    # Timer: simulation
    time_simulation_start = MPI.Wtime()

    # Run simulation
    if settings.eigenvalue_mode:
        eigenvalue_simulation(mcdc_arr, data)
    else:
        fixed_source_simulation(mcdc_arr, data)

    # Timer: simulation
    time_simulation_end = MPI.Wtime()

    # ==================================================================================
    # Working on the output
    # ==================================================================================

    # Timer: output
    time_output_start = MPI.Wtime()

    # Compressed sensing reconstruction
    '''
    N_cs_bins = mcdc["cs_tallies"]["filter"]["N_cs_bins"][0]
    if N_cs_bins != 0:
        cs_reconstruct(mcdc)
    '''

    # Generate hdf5 output files
    generate_hdf5(mcdc, data)

    # Timer: output
    time_output_end = MPI.Wtime()

    # ==================================================================================
    # Finalizing
    # ==================================================================================

    # Final barrier
    MPI.COMM_WORLD.Barrier()

    # Timer: total
    time_total_end = MPI.Wtime()

    # Manage timers
    mcdc["runtime_total"] = time_total_end - time_total_start
    mcdc["runtime_preparation"] = time_prep_end - time_prep_start
    mcdc["runtime_simulation"] = time_simulation_end - time_simulation_start
    mcdc["runtime_output"] = time_output_end - time_output_start
    if master:
        save_runtime(mcdc)
        print_runtime(mcdc)

    # GPU closeout
    teardown_gpu(mcdc)

    input_deck.reset()


# ======================================================================================
# Prepare
# ======================================================================================


def prepare():
    # =========================================================================
    # Simulation settings
    # =========================================================================

    # Get settings
    settings = simulation.settings

    # Set physics mode
    settings.multigroup_mode = isinstance(
        simulation.materials[0], MaterialMG
    )
    
    # Reset time grid size of all tallies if census-based tally is desired
    if settings.use_census_based_tally:
        N_bin = settings.census_tally_frequency
        for tally in simulation.tallies:
            tally._use_census_based_tally(N_bin)
    
    # Set appropriate time boundary
    settings.time_boundary = max(
        [settings.time_boundary] + [tally.time[-1] for tally in simulation.tallies]
    )
    
    # =========================================================================
    # Simulation parameters
    # =========================================================================

    # Create root universe if not defined
    if len(simulation.universes[0].cells) == 0:
        simulation.universes[0].cells = simulation.cells

    # Initial guess
    simulation.k_eff = settings.k_init

    # Activate tally scoring for fixed-source
    if not settings.eigenvalue_mode:
        simulation.cycle_active = True
    # All active eigenvalue cycle?
    elif settings.N_inactive == 0:
        simulation.cycle_active = True

    # ==================================================================================
    # Set particle bank sizes
    # ==================================================================================

    # Some sizes
    N_particle = settings.N_particle
    N_work = math.ceil(N_particle / MPI.COMM_WORLD.Get_size())
    N_census = settings.N_census

    # Determine bank size
    size_active = 1 + settings.active_bank_buffer
    size_census = 0
    size_source = 0
    size_future = 0
    #
    if settings.eigenvalue_mode or N_census > 1:
        size_census = 1 + int((1 + settings.census_bank_buffer_ratio) * N_work)
        size_source = 1 + int((1 + settings.source_bank_buffer_ratio) * N_work)
    if N_census > 1:
        size_future = 1 + int((1 + settings.future_bank_buffer_ratio) * N_work)

    # Set bank size
    simulation.bank_active.size[0] = size_active
    simulation.bank_census.size[0] = size_census
    simulation.bank_source.size[0] = size_source
    simulation.bank_future.size[0] = size_future

    # ==================================================================================
    # Generate Numba-supported "Objects"
    # ==================================================================================
    
    mcdc_arr, data = code_factory.generate_numba_objects(simulation)

    # =========================================================================
    # Platform Targeting, Adapters, Toggles, etc
    # =========================================================================

    # Adapt kernels
    kernel.adapt_rng(nb.config.DISABLE_JIT)
    type_.make_size_rpn(simulation.cells)
    settings.target_gpu = True if config.target == "gpu" else False

    if config.target == "gpu":
        if MPI.COMM_WORLD.Get_rank() != 0:
            adapt.harm.config.should_compile(adapt.harm.config.ShouldCompile.NEVER)
        elif config.caching == False:
            adapt.harm.config.should_compile(adapt.harm.config.ShouldCompile.ALWAYS)
        if not adapt.HAS_HARMONIZE:
            print_error(
                "No module named 'harmonize' - GPU functionality not available. "
            )
        adapt.gpu_forward_declare(config.args)

    # adapt.set_toggle("iQMC", input_deck.technique["iQMC"])
    adapt.set_toggle("domain_decomp", input_deck.technique["domain_decomposition"])
    adapt.eval_toggle()
    adapt.target_for(config.target)
    if config.target == "gpu":
        build_gpu_progs(input_deck, config.args)
    adapt.nopython_mode((config.mode == "numba") or (config.mode == "numba_debug"))

    # =========================================================================
    # Particle banks
    # =========================================================================

    # Particle bank tags
    mcdc["bank_active"]["tag"] = "active"
    mcdc["bank_census"]["tag"] = "census"
    mcdc["bank_source"]["tag"] = "source"
    mcdc["bank_future"]["tag"] = "future"

    # =========================================================================
    # Source file
    #   TODO: Use parallel h5py
    # =========================================================================

    # All ranks, take turn
    for i in range(mcdc["mpi_size"]):
        if mcdc["mpi_rank"] == i:
            if settings.use_source_file:
                with h5py.File(settings.source_file_name, "r") as f:
                    # Get source particle size
                    N_particle = f["particles_size"][()]

                    # Redistribute work
                    kernel.distribute_work(N_particle, mcdc)
                    N_local = mcdc["mpi_work_size"]
                    start = mcdc["mpi_work_start"]
                    end = start + N_local

                    # Add particles to source bank
                    mcdc["bank_source"]["particles"][:N_local] = f["particles"][
                        start:end
                    ]
                    mcdc["bank_source"]["size"] = N_local
        MPI.COMM_WORLD.Barrier()

    # =========================================================================
    # Finalize data: wrapping into a tuple
    # =========================================================================

    setup_gpu(mcdc)

    # Pick physics model
    if settings.multigroup_mode:
        physics.neutron.particle_speed = physics.neutron.multigroup.particle_speed
        physics.neutron.macro_xs = physics.neutron.multigroup.macro_xs
        physics.neutron.neutron_production_xs = (
            physics.neutron.multigroup.neutron_production_xs
        )
        physics.neutron.collision = physics.neutron.multigroup.collision

    # TODO: Delete Python objects if running in Numba mode

    return mcdc_arr, data


def calculate_cs_A(data, mcdc):
    x_grid = mcdc["mesh_tallies"]["filter"]["x"][0]
    y_grid = mcdc["mesh_tallies"]["filter"]["y"][0]
    Nx = len(x_grid) - 1
    Ny = len(y_grid) - 1

    N_cs_bins = mcdc["cs_tallies"]["filter"]["N_cs_bins"][0]
    cs_bin_size = mcdc["cs_tallies"]["filter"]["cs_bin_size"][0]

    S = [[] for _ in range(N_cs_bins)]

    [x_centers, y_centers] = mcdc["cs_tallies"]["filter"]["cs_centers"][0]
    x_centers[-1] = (x_grid[-1] + x_grid[0]) / 2
    y_centers[-1] = (y_grid[-1] + y_grid[0]) / 2

    # Calculate the overlap grid for each bin, and flatten into a row of S
    for ibin in range(N_cs_bins):
        if ibin == N_cs_bins - 1:
            # could just change to -INF, INF
            cs_bin_size = np.array([x_grid[-1] + x_grid[0], y_grid[-1] + y_grid[0]])

        bin_x_min = x_centers[ibin] - cs_bin_size[0] / 2
        bin_x_max = x_centers[ibin] + cs_bin_size[0] / 2
        bin_y_min = y_centers[ibin] - cs_bin_size[1] / 2
        bin_y_max = y_centers[ibin] + cs_bin_size[1] / 2

        overlap = np.zeros((len(y_grid) - 1, len(x_grid) - 1))

        for i in range(len(y_grid) - 1):
            for j in range(len(x_grid) - 1):
                cell_x_min = x_grid[j]
                cell_x_max = x_grid[j + 1]
                cell_y_min = y_grid[i]
                cell_y_max = y_grid[i + 1]

                # Calculate overlap in x and y directions
                overlap_x = np.maximum(
                    0,
                    np.minimum(bin_x_max, cell_x_max)
                    - np.maximum(bin_x_min, cell_x_min),
                )
                overlap_y = np.maximum(
                    0,
                    np.minimum(bin_y_max, cell_y_max)
                    - np.maximum(bin_y_min, cell_y_min),
                )

                # Calculate fractional overlap
                cell_area = (cell_x_max - cell_x_min) * (cell_y_max - cell_y_min)
                overlap[i, j] = (overlap_x * overlap_y) / cell_area

        S[ibin] = overlap.flatten()
    S = np.array(S)
    mcdc["cs_tallies"]["filter"]["cs_S"] = S

    assert np.allclose(S[-1], np.ones(Nx * Ny)), "Last row of S must be all ones"
    assert S.shape[1] == Nx * Ny, "Size of S must match Nx * Ny."
    assert (
        S.shape[1] == mcdc["cs_tallies"]["N_bin"][0]
    ), "Size of S must match number of cells in desired mesh tally"

    # TODO: can this be done in a different way? idk
    # Construct the DCT matrix T
    idct_basis_x = spfft.idct(np.identity(Nx), axis=0)
    idct_basis_y = spfft.idct(np.identity(Ny), axis=0)

    T_inv = np.kron(idct_basis_y, idct_basis_x)
    A = S @ T_inv
    return A, T_inv


def calculate_cs_sparse_solution(data, mcdc, A, b):
    N_fine_cells = mcdc["cs_tallies"]["N_bin"][0]

    # setting up the problem with CVXPY
    vx = cp.Variable(N_fine_cells)

    # Basis pursuit denoising
    l = 0.5
    objective = cp.Minimize(0.5 * cp.norm(A @ vx - b, 2) + l * cp.norm(vx, 1))
    prob = cp.Problem(objective)
    result = prob.solve(verbose=False)

    # # Basis pursuit
    # objective = cp.Minimize(cp.norm(vx, 1))
    # constraints = [A @ vx == b]
    # prob = cp.Problem(objective, constraints)
    # result = prob.solve(verbose=True)
    # print(f'vx.value = {vx.value}')

    # formatting the sparse solution
    sparse_solution = np.array(vx.value).squeeze()

    return sparse_solution


def cs_reconstruct(data, mcdc):
    tally_bin = data
    tally = mcdc["cs_tallies"][0]
    stride = tally["stride"]
    bin_idx = stride["tally"]
    N_cs_bins = tally["filter"]["N_cs_bins"]
    Nx = len(mcdc["mesh_tallies"]["filter"]["x"][0]) - 1
    Ny = len(mcdc["mesh_tallies"]["filter"]["y"][0]) - 1

    b = tally_bin[TALLY_SUM, bin_idx : bin_idx + N_cs_bins]

    A, T_inv = calculate_cs_A(data, mcdc)
    x = calculate_cs_sparse_solution(data, mcdc, A, b)

    recon = T_inv @ x
    recon_reshaped = recon.reshape(Ny, Nx)

    tally["filter"]["cs_reconstruction"] = recon_reshaped


# =============================================================================
# utilities for handling discrepancies between input and program types
# =============================================================================


def copy_field(dst, src, name):
    if "padding" in name:
        return

    if isinstance(src, dict):
        data = src[name]
    else:
        data = getattr(src, name)

    if isinstance(dst[name], np.ndarray):
        if isinstance(data, np.ndarray) and dst[name].shape != data.shape:
            for dim in data.shape:
                if dim == 0:
                    return
            print(
                f"Warning: Dimension mismatch between input deck and global state for field '{name}'."
            )
            print(
                f"State dimension {dst[name].shape} does not match input dimension {src[name].shape}"
            )
        elif isinstance(data, list) and dst[name].shape[0] != len(data):
            if len(src[name]) == 0:
                return
            print(
                f"Warning: Dimension mismatch between input deck and global state for field '{name}'."
            )
            print(
                f"State dimension {dst[name].shape} does not match input dimension {len(src[name])}"
            )

    dst[name] = data


def cardlist_to_h5group(dictlist, input_group, name):
    if name[-1] != "s":
        main_group = input_group.create_group(name + "s")
    else:
        main_group = input_group.create_group(name)
    for item in dictlist:
        if "ID" in dir(item):
            group = main_group.create_group(name + "_%i" % getattr(item, "ID"))
        card_to_h5group(item, group)


def card_to_h5group(card, group):
    for name in [
        a
        for a in dir(card)
        if not a.startswith("__") and not callable(getattr(card, a)) and a != "tag"
    ]:
        value = getattr(card, name)
        if type(value) == dict:
            dict_to_h5group(value, group.create_group(name))
        elif value is None:
            next
        else:
            if name not in ["region"]:
                group[name] = value

            elif name == "region":
                group[name] = str(value)


def dictlist_to_h5group(dictlist, input_group, name):
    if name[-1] != "s":
        main_group = input_group.create_group(name + "s")
    else:
        main_group = input_group.create_group(name)
    for item in dictlist:
        group = main_group.create_group(name + "_%i" % item["ID"])
        dict_to_h5group(item, group)


def dict_to_h5group(dict_, group):
    for k, v in dict_.items():
        if type(v) == dict:
            dict_to_h5group(dict_[k], group.create_group(k))
        elif v is None:
            next
        else:
            group[k] = v


def dd_mergetally(mcdc, data_tally):
    """
    Performs tally recombination on domain-decomposed mesh tallies.
    Gathers and re-organizes tally data into a single array as it
      would appear in a non-decomposed simulation.
    """

    # create bin for recomposed tallies
    d_Nx = input_deck.technique["dd_mesh"]["x"].size - 1
    d_Ny = input_deck.technique["dd_mesh"]["y"].size - 1
    d_Nz = input_deck.technique["dd_mesh"]["z"].size - 1

    # capture tally lengths for reorganizing later
    xlen = mcdc["technique"]["dd_xlen"]
    ylen = mcdc["technique"]["dd_ylen"]
    zlen = mcdc["technique"]["dd_zlen"]

    # MPI gather
    if (d_Nx * d_Ny * d_Nz) == MPI.COMM_WORLD.Get_size():
        sendcounts = np.array(MPI.COMM_WORLD.gather(len(data_tally[0]), root=0))
        if mcdc["mpi_master"]:
            dd_tally = np.zeros((data_tally.shape[0], sum(sendcounts)))
        else:
            dd_tally = np.empty(data_tally.shape[0])  # dummy tally
        # gather tallies
        for i, t in enumerate(data_tally):
            MPI.COMM_WORLD.Gatherv(
                sendbuf=data_tally[i], recvbuf=(dd_tally[i], sendcounts), root=0
            )
        # gather tally lengths for proper recombination
        xlens = MPI.COMM_WORLD.gather(xlen, root=0)
        ylens = MPI.COMM_WORLD.gather(ylen, root=0)
        zlens = MPI.COMM_WORLD.gather(zlen, root=0)

    # MPI gather for multiprocessor subdomains
    else:
        i = 0
        dd_ranks = []
        # find nonzero tally processor IDs
        for n in range(d_Nx * d_Ny * d_Nz):
            dd_ranks.append(i)
            i += int(mcdc["technique"]["dd_work_ratio"][n])
        # create MPI comm group for nonzero tallies
        dd_group = MPI.COMM_WORLD.group.Incl(dd_ranks)
        dd_comm = MPI.COMM_WORLD.Create(dd_group)
        dd_tally = np.empty(data_tally.shape[0])  # dummy tally

        if MPI.COMM_NULL != dd_comm:
            sendcounts = np.array(dd_comm.gather(len(data_tally[0]), root=0))
            if mcdc["mpi_master"]:
                dd_tally = np.zeros((data_tally.shape[0], sum(sendcounts)))
            # gather tallies
            for i, t in enumerate(data_tally):
                dd_comm.Gatherv(data_tally[i], (dd_tally[i], sendcounts), root=0)
            # gather tally lengths for proper recombination
            xlens = dd_comm.gather(xlen, root=0)
            ylens = dd_comm.gather(ylen, root=0)
            zlens = dd_comm.gather(zlen, root=0)
        dd_group.Free()
        if MPI.COMM_NULL != dd_comm:
            dd_comm.Free()

    if mcdc["mpi_master"]:
        buff = np.zeros_like(dd_tally)
        # reorganize tally data
        # TODO: find/develop a more efficient algorithm for this
        tally_idx = 0
        offset = 0
        ysum = mcdc["technique"]["dd_ysum"]
        zsum = mcdc["technique"]["dd_zsum"]
        for di in range(0, d_Nx * d_Ny * d_Nz):
            dz = di // (d_Nx * d_Ny)
            dy = (di % (d_Nx * d_Ny)) // d_Nx
            dx = di % d_Nx

            offset = 0
            # calculate subdomain offset
            for i in range(0, dx):
                offset += xlens[i] * ysum * zsum

            for i in range(0, dy):
                y_ind = i * d_Nx
                offset += ylens[y_ind] * zsum

            for i in range(0, dz):
                z_ind = i * d_Nx * d_Ny
                offset += zlens[z_ind]

            # calculate index within subdomain
            xlen = xlens[di]
            ylen = ylens[di]
            zlen = zlens[di]
            for xi in range(0, xlen):
                for yi in range(0, ylen):
                    for zi in range(0, zlen):
                        # calculate reorganized index
                        ind_x = xi * ysum * zsum
                        ind_y = yi * zsum
                        ind_z = zi
                        buff_idx = offset + ind_x + ind_y + ind_z
                        # place tally value in correct position
                        buff[:, buff_idx] = dd_tally[:, tally_idx]
                        tally_idx += 1
        # replace old tally with reorganized tally
        dd_tally = buff

    return dd_tally


def dd_mergemesh(mcdc, data_tally):
    """
    Performs mesh recombination on domain-decomposed mesh tallies.
    Gathers and re-organizes mesh data into a single array as it
      would appear in a non-decomposed simulation.
    """
    d_Nx = input_deck.technique["dd_mesh"]["x"].size - 1
    d_Ny = input_deck.technique["dd_mesh"]["y"].size - 1
    d_Nz = input_deck.technique["dd_mesh"]["z"].size - 1
    # gather mesh filter
    if d_Nx > 1:
        sendcounts = np.array(
            MPI.COMM_WORLD.gather(len(mcdc["mesh_tallies"][0]["filter"]["x"]), root=0)
        )
        if mcdc["mpi_master"]:
            x_filter = np.zeros((mcdc["mesh_tallies"].shape[0], sum(sendcounts)))
        else:
            x_filter = np.empty((mcdc["mesh_tallies"].shape[0]))  # dummy tally
        # gather mesh
        for i in range(mcdc["mesh_tallies"].shape[0]):
            MPI.COMM_WORLD.Gatherv(
                sendbuf=mcdc["mesh_tallies"][i]["filter"]["x"],
                recvbuf=(x_filter[i], sendcounts),
                root=0,
            )
        if mcdc["mpi_master"]:
            x_final = np.zeros((mcdc["mesh_tallies"].shape[0], x_filter.shape[1] + 1))
            x_final[:, 0] = mcdc["mesh_tallies"][:]["filter"]["x"][0][0]
            x_final[:, 1:] = x_filter

    if d_Ny > 1:
        sendcounts = np.array(
            MPI.COMM_WORLD.gather(len(mcdc["mesh_tallies"][0]["filter"]["y"]), root=0)
        )
        if mcdc["mpi_master"]:
            y_filter = np.zeros((mcdc["mesh_tallies"].shape[0], sum(sendcounts)))
        else:
            y_filter = np.empty((mcdc["mesh_tallies"].shape[0]))  # dummy tally
        # gather mesh
        for i in range(mcdc["mesh_tallies"].shape[0]):
            MPI.COMM_WORLD.Gatherv(
                sendbuf=mcdc["mesh_tallies"][i]["filter"]["y"],
                recvbuf=(y_filter[i], sendcounts),
                root=0,
            )
        if mcdc["mpi_master"]:
            y_final = np.zeros((mcdc["mesh_tallies"].shape[0], y_filter.shape[1] + 1))
            y_final[:, 0] = mcdc["mesh_tallies"][:]["filter"]["y"][0][0]
            y_final[:, 1:] = y_filter

    if d_Nz > 1:
        sendcounts = np.array(
            MPI.COMM_WORLD.gather(
                len(mcdc["mesh_tallies"][0]["filter"]["z"]) - 1, root=0
            )
        )
        if mcdc["mpi_master"]:
            z_filter = np.zeros((mcdc["mesh_tallies"].shape[0], sum(sendcounts)))
        else:
            z_filter = np.empty((mcdc["mesh_tallies"].shape[0]))  # dummy tally
        # gather mesh
        for i in range(mcdc["mesh_tallies"].shape[0]):
            MPI.COMM_WORLD.Gatherv(
                sendbuf=mcdc["mesh_tallies"][i]["filter"]["z"][1:],
                recvbuf=(z_filter[i], sendcounts),
                root=0,
            )
        if mcdc["mpi_master"]:
            z_final = np.zeros((mcdc["mesh_tallies"].shape[0], z_filter.shape[1] + 1))
            z_final[:, 0] = mcdc["mesh_tallies"][:]["filter"]["z"][0][0]
            z_final[:, 1:] = z_filter

    dd_mesh = []
    if mcdc["mpi_master"]:
        if d_Nx > 1:
            dd_mesh.append(x_final)
        else:
            dd_mesh.append(mcdc["mesh_tallies"][:]["filter"]["x"])
        if d_Ny > 1:
            dd_mesh.append(y_final)
        else:
            dd_mesh.append(mcdc["mesh_tallies"][:]["filter"]["y"])
        if d_Nz > 1:
            dd_mesh.append(z_final)
        else:
            dd_mesh.append(mcdc["mesh_tallies"][:]["filter"]["z"])
    return dd_mesh


def generate_hdf5(mcdc, data):

    if mcdc["technique"]["domain_decomposition"]:
        dd_tally = dd_mergetally(mcdc, data_tally)
        dd_mesh = dd_mergemesh(mcdc, data_tally)

    if mcdc["mpi_master"]:
        if mcdc["settings"]["use_progress_bar"]:
            print_msg("")
        print_msg(" Generating output HDF5 files...")

        with h5py.File(mcdc["settings"]["output_name"] + ".h5", "w") as f:
            # Version
            version = importlib.metadata.version("mcdc")
            f["version"] = version

            # Input deck
            if mcdc["settings"]["save_input_deck"]:
                input_group = f.create_group("input_deck")
                # cardlist_to_h5group(simulation.nuclides, input_group, "nuclide")
                # cardlist_to_h5group(simulation.materials, input_group, "material")
                # cardlist_to_h5group(input_deck.surfaces, input_group, "surface")
                # cardlist_to_h5group(input_deck.cells, input_group, "cell")
                # cardlist_to_h5group(input_deck.universes, input_group, "universe")
                # cardlist_to_h5group(input_deck.lattices, input_group, "lattice")
                cardlist_to_h5group(input_deck.sources, input_group, "source")
                #cardlist_to_h5group(
                #    input_deck.mesh_tallies, input_group, "mesh_tallies"
                #)
                #cardlist_to_h5group(
                #    input_deck.surface_tallies, input_group, "surface_tallies"
                #)
                #cardlist_to_h5group(
                #    input_deck.cell_tallies, input_group, "cell_tallies"
                #)
                #cardlist_to_h5group(input_deck.cs_tallies, input_group, "cs_tallies")
                #card_to_h5group(
                #    simulation.settings, input_group.create_group("setting")
                #)
                dict_to_h5group(
                    input_deck.technique, input_group.create_group("technique")
                )

            # No need to output tally if time census-based tally is used
            if mcdc["settings"]["use_census_based_tally"]:
                return

            # Cell and surface tallies
            for tally in mcdc['cell_tallys']:
                tally_name = tally['name']

                # Filter grids
                f.create_dataset(f"tallies/{tally_name}/grid/mu", data=mcdc_get.tally.mu_all(tally, data))
                f.create_dataset(f"tallies/{tally_name}/grid/azi", data=mcdc_get.tally.azi_all(tally, data))
                f.create_dataset(f"tallies/{tally_name}/grid/energy", data=mcdc_get.tally.energy_all(tally, data))
                f.create_dataset(f"tallies/{tally_name}/grid/time", data=mcdc_get.tally.time_all(tally, data))

                # Shape
                N_mu = tally["mu_length"] - 1
                N_azi = tally["azi_length"] - 1
                N_energy = tally["energy_length"] - 1
                N_time = tally["time_length"] - 1
                N_score = tally["scores_length"]
                shape = (N_mu, N_azi, N_energy, N_time, N_score)

                # Reshape tally
                N_bin = tally['bin_length']
                start_mean = tally['bin_sum_offset']
                start_sdev = tally['bin_sum_square_offset']
                mean = data[start_mean:start_mean+N_bin]
                sdev = data[start_sdev:start_sdev+N_bin]
                mean = mean.reshape(shape)
                sdev = sdev.reshape(shape)

                # Roll tally so that score is in the front
                mean = np.rollaxis(mean, 4, 0)
                sdev = np.rollaxis(sdev, 4, 0)

                # Iterate over scores
                for i in range(tally['scores_length']):
                    score_type = mcdc_get.tally.scores(i, tally, data)
                    score_mean = np.squeeze(mean[i])
                    score_sdev = np.squeeze(sdev[i])
                    if score_type == SCORE_FLUX:
                        score_name = "flux"
                    group_name = f"tallies/{tally_name}/{score_name}/"

                    f.create_dataset(group_name + "mean", data=score_mean)
                    f.create_dataset(group_name + "sdev", data=score_sdev)
                    if mcdc["technique"]["uq"]:
                        mc_var = score_tally_bin[TALLY_UQ_BATCH_VAR]
                        tot_var = score_tally_bin[TALLY_UQ_BATCH]
                        uq_var = tot_var - mc_var
                        f.create_dataset(group_name + "uq_var", data=uq_var)

            # Mesh tallies
            for tally in mcdc['mesh_tallys']:
                tally_name = tally['name']

                # Get mesh
                mesh_type = tally['mesh_type']
                mesh_ID = tally['mesh_ID']
                if mesh_type == MESH_UNIFORM:
                    mesh = mcdc['uniform_meshs'][mesh_ID]
                elif mesh_type == MESH_STRUCTURED:
                    mesh = mcdc['structured_meshs'][mesh_ID]

                # Filter grids
                f.create_dataset(f"tallies/{tally_name}/grid/mu", data=mcdc_get.tally.mu_all(tally, data))
                f.create_dataset(f"tallies/{tally_name}/grid/azi", data=mcdc_get.tally.azi_all(tally, data))
                f.create_dataset(f"tallies/{tally_name}/grid/energy", data=mcdc_get.tally.energy_all(tally, data))
                f.create_dataset(f"tallies/{tally_name}/grid/t", data=mcdc_get.tally.time_all(tally, data))
                f.create_dataset(f"tallies/{tally_name}/grid/x", data=mcdc_get.structured_mesh.x_all(mesh, data))
                f.create_dataset(f"tallies/{tally_name}/grid/y", data=mcdc_get.structured_mesh.y_all(mesh, data))
                f.create_dataset(f"tallies/{tally_name}/grid/z", data=mcdc_get.structured_mesh.z_all(mesh, data))

                # Shape
                N_mu = tally["mu_length"] - 1
                N_azi = tally["azi_length"] - 1
                N_energy = tally["energy_length"] - 1
                N_t = tally["time_length"] - 1
                N_x = mesh["Nx"]
                N_y = mesh["Ny"]
                N_z = mesh["Nz"]
                N_score = tally["scores_length"]
                shape = (N_mu, N_azi, N_energy, N_t, N_x, N_y, N_z, N_score)

                # Reshape tally
                N_bin = tally['bin_length']
                start_mean = tally['bin_sum_offset']
                start_sdev = tally['bin_sum_square_offset']
                mean = data[start_mean:start_mean+N_bin]
                sdev = data[start_sdev:start_sdev+N_bin]
                mean = mean.reshape(shape)
                sdev = sdev.reshape(shape)

                # Roll tally so that score is in the front
                mean = np.rollaxis(mean, 7, 0)
                sdev = np.rollaxis(sdev, 7, 0)

                # Iterate over scores
                for i in range(tally['scores_length']):
                    score_type = mcdc_get.tally.scores(i, tally, data)
                    score_mean = np.squeeze(mean[i])
                    score_sdev = np.squeeze(sdev[i])
                    if score_type == SCORE_FLUX:
                        score_name = "flux"
                    group_name = f"tallies/{tally_name}/{score_name}/"

                    f.create_dataset(group_name + "mean", data=score_mean)
                    f.create_dataset(group_name + "sdev", data=score_sdev)
                    if mcdc["technique"]["uq"]:
                        mc_var = score_tally_bin[TALLY_UQ_BATCH_VAR]
                        tot_var = score_tally_bin[TALLY_UQ_BATCH]
                        uq_var = tot_var - mc_var
                        f.create_dataset(group_name + "uq_var", data=uq_var)

            # Eigenvalues
            if mcdc["settings"]["eigenvalue_mode"]:
                if mcdc["technique"]["iQMC"]:
                    f.create_dataset("k_eff", data=mcdc["k_eff"])
                    if mcdc["technique"]["iqmc"]["mode"] == "batched":
                        N_cycle = mcdc["setting"]["N_cycle"]
                        f.create_dataset("k_cycle", data=mcdc["k_cycle"][:N_cycle])
                        f.create_dataset("k_mean", data=mcdc["k_avg_running"])
                        f.create_dataset("k_sdev", data=mcdc["k_sdv_running"])
                else:
                    N_cycle = mcdc["settings"]["N_cycle"]
                    f.create_dataset("k_cycle", data=mcdc["k_cycle"][:N_cycle])
                    f.create_dataset("k_mean", data=mcdc["k_avg_running"])
                    f.create_dataset("k_sdev", data=mcdc["k_sdv_running"])
                    f.create_dataset("global_tally/neutron/mean", data=mcdc["n_avg"])
                    f.create_dataset("global_tally/neutron/sdev", data=mcdc["n_sdv"])
                    f.create_dataset("global_tally/neutron/max", data=mcdc["n_max"])
                    f.create_dataset("global_tally/precursor/mean", data=mcdc["C_avg"])
                    f.create_dataset("global_tally/precursor/sdev", data=mcdc["C_sdv"])
                    f.create_dataset("global_tally/precursor/max", data=mcdc["C_max"])
                    if mcdc["settings"]["use_gyration_radius"]:
                        f.create_dataset(
                            "gyration_radius", data=mcdc["gyration_radius"][:N_cycle]
                        )

            # iQMC
            if mcdc["technique"]["iQMC"]:
                # iQMC mesh
                T = mcdc["technique"]
                f.create_dataset("iqmc/grid/t", data=T["iqmc"]["mesh"]["t"])
                f.create_dataset("iqmc/grid/x", data=T["iqmc"]["mesh"]["x"])
                f.create_dataset("iqmc/grid/y", data=T["iqmc"]["mesh"]["y"])
                f.create_dataset("iqmc/grid/z", data=T["iqmc"]["mesh"]["z"])
                # Scores
                for name in [
                    "flux",
                    "source-x",
                    "source-y",
                    "source-z",
                    "fission-power",
                ]:
                    if T["iqmc"]["score_list"][name]:
                        name_h5 = name.replace("-", "_")
                        f.create_dataset(
                            f"iqmc/tally/{name_h5}/mean",
                            data=np.squeeze(T["iqmc"]["score"][name]["mean"]),
                        )
                        f.create_dataset(
                            f"iqmc/tally/{name_h5}/sdev",
                            data=np.squeeze(T["iqmc"]["score"][name]["sdev"]),
                        )
                # iQMC source strength
                f.create_dataset(
                    "iqmc/tally/source_constant/mean",
                    data=np.squeeze(T["iqmc"]["source"]),
                )
                # Iteration data
                f.create_dataset(
                    "iqmc/iteration_count", data=T["iqmc"]["iteration_count"]
                )
                f.create_dataset("iqmc/sweep_count", data=T["iqmc"]["sweep_count"])
                f.create_dataset("iqmc/final_residual", data=T["iqmc"]["residual"])

    # Save particle?
    if mcdc["settings"]["save_particle"]:
        # Gather source bank
        # TODO: Parallel HDF5 and mitigation of large data passing
        N = mcdc["bank_source"]["size"][0]
        neutrons = MPI.COMM_WORLD.gather(mcdc["bank_source"]["particles"][:N])

        # Master saves the particle
        if mcdc["mpi_master"]:
            # Remove unwanted particle fields
            neutrons = np.concatenate(neutrons[:])

            # Create dataset
            with h5py.File(mcdc["setting"]["output_name"] + ".h5", "a") as f:
                f.create_dataset("particles", data=neutrons[:])
                f.create_dataset("particles_size", data=len(neutrons[:]))


def replace_dataset(file_, field, data):
    if field in file_:
        del file_[field]
    file_.create_dataset(field, data=data)


def recombine_tallies(file="output.h5"):
    if MPI.COMM_WORLD.Get_rank() == 0:
        # Load main output file and read input params
        with h5py.File(file, "r") as f:
            output_name = str(f["input_deck/setting/output_name"][()])[2:-1]
            N_particle = f["input_deck/setting/N_particle"][()]
            N_census = f["input_deck/setting/N_census"][()] - 1
            N_batch = f["input_deck/setting/N_batch"][()]
            N_frequency = f["input_deck/setting/census_tally_frequency"][()]
        Nt = N_census * N_frequency
        # Combine the tally output into a single file

        collected_tallies = []
        collected_tally_names = []
        # Collecting info on number and types of tallies
        for i_census in range(N_census):
            for i_batch in range(N_batch):
                with h5py.File(
                    output_name + "-batch_%i-census_%i.h5" % (i_batch, i_census), "r"
                ) as f:
                    tallies = f["tallies"]
                    for tally in tallies:
                        if tally not in collected_tally_names:
                            grid = tallies[tally]["grid"]
                            tally_list = [tally]
                            for tally_type in tallies[tally]:
                                if tally_type != "grid":
                                    tally_list.append(tally_type)
                            collected_tallies.append(tally_list)
                            collected_tally_names.append(tally)

        for i, tally_info in enumerate(collected_tallies):
            tally_type = tally_info[0].split("_")[0]
            tally_number = tally_info[0].split("_")[-1]
            with h5py.File(output_name + ".h5", "a") as f:
                grid = f[
                    "input_deck/"
                    + tally_type
                    + "_tallies/"
                    + tally_type
                    + "_tallies_"
                    + tally_number
                ]
                t_final = f["input_deck/setting/census_time"][()][-2]
                t = np.linspace(0, t_final, N_census * N_frequency + 1)
                Nx = len(grid["x"][()]) - 1
                Ny = len(grid["y"][()]) - 1
                Nz = len(grid["z"][()]) - 1
                Nmu = len(grid["mu"][()]) - 1
                N_azi = len(grid["azi"][()]) - 1
                Ng = len(grid["g"][()]) - 1

                # Creating structure of correct size to hold combined tally
                for tally_type in tally_info[1:]:
                    tally_score = np.zeros((Nt, Nmu, N_azi, Ng, Nx, Ny, Nz))
                    tally_score = np.squeeze(tally_score)
                    tally_score_sq = np.zeros_like(tally_score)

                    # Number of shift of time index
                    N_shift = 0
                    if Nmu > 1:
                        N_shift += 1
                    if N_azi > 1:
                        N_shift += 1
                    if Ng > 1:
                        N_shift += 1

                    for i_census in range(N_census):
                        idx_start = i_census * N_frequency
                        idx_end = idx_start + N_frequency
                        for i_batch in range(N_batch):
                            with h5py.File(
                                output_name
                                + "-batch_%i-census_%i.h5" % (i_batch, i_census),
                                "r",
                            ) as f1:
                                score = f1[
                                    "tallies/"
                                    + tally_info[0]
                                    + "/"
                                    + tally_type
                                    + "/score"
                                ][:]
                                if N_shift > 0:
                                    score = np.rollaxis(score, N_shift, 0)
                                tally_score[idx_start:idx_end] += score
                                tally_score_sq[idx_start:idx_end] += score * score
                    tally_score /= N_batch
                    if N_batch > 0:
                        tally_score_sq = np.sqrt(
                            (tally_score_sq / N_batch - np.square(tally_score))
                            / (N_batch - 1)
                        )

                    field_base = "tallies/" + tally_info[0] + "/" + tally_type
                    replace_dataset(f, field_base + "/mean", tally_score)
                    replace_dataset(f, field_base + "/sdev", tally_score_sq)

                replace_dataset(
                    f, "tallies/" + tally_info[0] + "/grid/x", grid["x"][()]
                )
                replace_dataset(
                    f, "tallies/" + tally_info[0] + "/grid/y", grid["y"][()]
                )
                replace_dataset(
                    f, "tallies/" + tally_info[0] + "/grid/z", grid["z"][()]
                )
                replace_dataset(
                    f, "tallies/" + tally_info[0] + "/grid/t", grid["t"][()]
                )
                replace_dataset(
                    f, "tallies/" + tally_info[0] + "/grid/mu", grid["mu"][()]
                )
                replace_dataset(
                    f, "tallies/" + tally_info[0] + "/grid/azi", grid["azi"][()]
                )
                replace_dataset(
                    f, "tallies/" + tally_info[0] + "/grid/g", grid["g"][()]
                )
        """
        for i_census in range(N_census):
            for i_batch in range(N_batch):
                file_name = (
                    output_name
                    + "-batch_"
                    + str(i_batch)
                    + "-census_"
                    + str(i_census)
                    + ".h5"
                )
                os.system("rm " + file_name)
        """


def save_runtime(mcdc):
    if mcdc["mpi_master"]:
        with h5py.File(mcdc["settings"]["output_name"] + ".h5", "a") as f:
            for name in [
                "total",
                "preparation",
                "simulation",
                "output",
                "bank_management",
            ]:
                f.create_dataset(
                    "runtime/" + name, data=np.array([mcdc["runtime_" + name]])
                )

        if config.args.runtime_output:
            with h5py.File(mcdc["setting"]["output_name"] + "-runtime.h5", "w") as f:
                for name in [
                    "total",
                    "preparation",
                    "simulation",
                    "output",
                    "bank_management",
                ]:
                    f.create_dataset(name, data=np.array([mcdc["runtime_" + name]]))


# ======================================================================================
# Visualize geometry
# ======================================================================================


def visualize(
    vis_type,
    x=0.0,
    y=0.0,
    z=0.0,
    pixels=(100, 100),
    colors=None,
    time=np.array([0.0]),
    save_as=None,
):
    """
    2D visualization of the created model

    Parameters
    ----------
    vis_plane : {'xy', 'yz', 'xz', 'zx', 'yz', 'zy'}
        Axis plane to visualize
    x : float or array_like
        Plane x-position (float) for 'yz' plot. Range of x-axis for 'xy' or 'xz' plot.
    y : float or array_like
        Plane y-position (float) for 'xz' plot. Range of y-axis for 'xy' or 'yz' plot.
    z : float or array_like
        Plane z-position (float) for 'xy' plot. Range of z-axis for 'xz' or 'yz' plot.
    time : array_like
        Times at which the geometry snapshots are taken
    pixels : array_like
        Number of respective pixels in the two axes in vis_plane
    colors : array_like
        List of pairs of material and its color
    """
    # TODO: add input error checkers

    _, mcdc_container = prepare()
    mcdc = mcdc_container[0]

    # Color assignment for materials (by material ID)
    if colors is not None:
        new_colors = {}
        for item in colors.items():
            new_colors[item[0].ID] = mpl_colors.to_rgb(item[1])
        colors = new_colors
    else:
        colors = {}
        for i in range(len(mcdc["materials"])):
            colors[i] = plt.cm.Set1(i)[:-1]
    WHITE = mpl_colors.to_rgb("white")

    # Set reference axis
    for axis in ["x", "y", "z"]:
        if axis not in vis_type:
            reference_key = axis

    if reference_key == "x":
        reference = x
    elif reference_key == "y":
        reference = y
    elif reference_key == "z":
        reference = z

    # Set first and second axes
    first_key = vis_type[0]
    second_key = vis_type[1]

    if first_key == "x":
        first = x
    elif first_key == "y":
        first = y
    elif first_key == "z":
        first = z

    if second_key == "x":
        second = x
    elif second_key == "y":
        second = y
    elif second_key == "z":
        second = z

    # Axis pixels sizes
    d_first = (first[1] - first[0]) / pixels[0]
    d_second = (second[1] - second[0]) / pixels[1]

    # Axis pixels grids and midpoints
    first_grid = np.linspace(first[0], first[1], pixels[0] + 1)
    first_midpoint = 0.5 * (first_grid[1:] + first_grid[:-1])

    second_grid = np.linspace(second[0], second[1], pixels[1] + 1)
    second_midpoint = 0.5 * (second_grid[1:] + second_grid[:-1])

    # Set dummy particle
    particle_container = adapt.local_array(1, type_.particle)
    particle = particle_container[0]
    particle[reference_key] = reference
    particle["g"] = 0
    particle["E"] = 1e6

    for t in time:
        # Set time
        particle["t"] = t

        # Random direction
        particle["ux"], particle["uy"], particle["uz"] = (
            kernel.sample_isotropic_direction(particle_container)
        )

        # RGB color data for each pixels
        data = np.zeros(pixels + (3,))

        # Loop over the two axes
        for i in range(pixels[0]):
            particle[first_key] = first_midpoint[i]
            for j in range(pixels[1]):
                particle[second_key] = second_midpoint[j]

                # Get material
                particle["cell_ID"] = -1
                particle["material_ID"] = -1
                if geometry.locate_particle(particle_container, mcdc):
                    data[i, j] = colors[particle["material_ID"]]
                else:
                    data[i, j] = WHITE

        data = np.transpose(data, (1, 0, 2))
        plt.imshow(data, origin="lower", extent=first + second)
        plt.xlabel(first_key + " [cm]")
        plt.ylabel(second_key + " [cm]")
        plt.title(reference_key + " = %.2f cm" % reference + ", time = %.2f s" % t)
        if save_as is not None:
            plt.savefig(save_as + "_%.2f.png" % t)
            plt.clf()
        else:
            plt.show()
