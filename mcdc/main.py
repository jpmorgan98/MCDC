# ======================================================================================
# Run
# ======================================================================================


def run():
    import mcdc.print_ as print_module
    from mpi4py import MPI

    # Timer: total
    time_total_start = MPI.Wtime()

    from mcdc.object_.simulation import simulation
    settings = simulation.settings
    master = MPI.COMM_WORLD.Get_rank() == 0

    # Override settings with command-line arguments
    import mcdc.config as config
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

    mcdc_arr, data = preparation()
    mcdc = mcdc_arr[0]

    # Print headers
    if master:
        print_module.print_banner()
        print_module.print_configuration()
        print(" Now running TNT...")
        if settings.eigenvalue_mode:
            print_module.print_eigenvalue_header(mcdc)

    # Timer: preparation
    time_prep_end = MPI.Wtime()

    # ==================================================================================
    # Running the simulation
    # ==================================================================================

    # Timer: simulation
    time_simulation_start = MPI.Wtime()

    # Run simulation
    import mcdc.transport.simulation as simulation_module
    if settings.eigenvalue_mode:
        simulation_module.eigenvalue_simulation(mcdc_arr, data)
    else:
        simulation_module.fixed_source_simulation(mcdc_arr, data)

    # Timer: simulation
    time_simulation_end = MPI.Wtime()

    # ==================================================================================
    # Working on the output
    # ==================================================================================
    
    import mcdc.transport.output as output_module

    # Timer: output
    time_output_start = MPI.Wtime()

    # Generate hdf5 output file
    output_module.generate_output(mcdc, data)

    # Timer: output
    time_output_end = MPI.Wtime()

    # Final barrier
    MPI.COMM_WORLD.Barrier()

    # Timer: total
    time_total_end = MPI.Wtime()

    # Manage timers
    mcdc["runtime_total"] = time_total_end - time_total_start
    mcdc["runtime_preparation"] = time_prep_end - time_prep_start
    mcdc["runtime_simulation"] = time_simulation_end - time_simulation_start
    mcdc["runtime_output"] = time_output_end - time_output_start
    output_module.create_runtime_datasets(mcdc)
    if master:
        print_module.print_runtime(mcdc)

    # ==================================================================================
    # Finalizing
    # ==================================================================================

    # GPU teardowns
    from mcdc.transport.simulation import teardown_gpu
    teardown_gpu(mcdc)


# ======================================================================================
# Preparation
# ======================================================================================


def preparation():
    import math

    from mpi4py import MPI

    from mcdc.object_.simulation import simulation
    from mcdc.object_.material import MaterialMG

    # ==================================================================================
    # Simulation settings
    # ==================================================================================

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
    settings.time_boundary = min(
        [settings.time_boundary] + [tally.time[-1] for tally in simulation.tallies]
    )
    
    # ==================================================================================
    # Simulation parameters
    # ==================================================================================

    # Normalize source probability
    norm = 0.0
    for source in simulation.sources:
        norm += source.probability
    for source in simulation.sources:
        source.probability /= norm

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
    if settings.eigenvalue_mode or N_census == 1:
        settings.future_bank_buffer_ratio = 0.0
    if not settings.eigenvalue_mode and N_census == 1:
        settings.census_bank_buffer_ratio = 0.0
        settings.source_bank_buffer_ratio = 0.0
    size_active = settings.active_bank_buffer
    size_census = int((settings.census_bank_buffer_ratio) * N_work)
    size_source = int((settings.source_bank_buffer_ratio) * N_work)
    size_future = int((settings.future_bank_buffer_ratio) * N_work)

    # Set bank size
    simulation.bank_active.size[0] = size_active
    simulation.bank_census.size[0] = size_census
    simulation.bank_source.size[0] = size_source
    simulation.bank_future.size[0] = size_future

    # ==================================================================================
    # Generate Numba-supported "Objects"
    # ==================================================================================
   
    import mcdc.code_factory.code_factory as code_factory
    mcdc_arr, data = code_factory.generate_numba_objects(simulation)
    mcdc = mcdc_arr[0]
    
    # Reload mcdc_get
    import importlib
    import mcdc.mcdc_get as mcdc_get
    importlib.reload(mcdc_get)

    # ==================================================================================
    # Platform Targeting, Adapters, Toggles, etc
    # ==================================================================================

    # Adapt kernels
    import numba as nb
    import mcdc.config as config
    import mcdc.transport.kernel as kernel
    code_factory.make_size_rpn(simulation.cells)
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

    from mcdc.code_factory.adapt import eval_toggle, target_for, nopython_mode
    eval_toggle()
    target_for(config.target)
    if config.target == "gpu":
        build_gpu_progs(input_deck, config.args)
    nopython_mode((config.mode == "numba") or (config.mode == "numba_debug"))

    # ==================================================================================
    # Source file
    #   TODO: Use parallel h5py
    # ==================================================================================

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

    # ==================================================================================
    # Finalize data: wrapping into a tuple
    # ==================================================================================

    from mcdc.transport.simulation import setup_gpu
    setup_gpu(mcdc)

    # Pick physics model
    import mcdc.transport.physics as physics
    if settings.multigroup_mode:
        physics.neutron.particle_speed = physics.neutron.multigroup.particle_speed
        physics.neutron.macro_xs = physics.neutron.multigroup.macro_xs
        physics.neutron.neutron_production_xs = (
            physics.neutron.multigroup.neutron_production_xs
        )
        physics.neutron.collision = physics.neutron.multigroup.collision

    # Pick Python-version RNG if needed
    import mcdc.transport.rng as rng
    if config.mode == 'python':
        rng.wrapping_add = rng.wrapping_add_python
        rng.wrapping_mul = rng.wrapping_mul_python

    # TODO: Delete Python objects if running in Numba mode

    return mcdc_arr, data


# ======================================================================================
# utilities for handling discrepancies between input and program types
# ======================================================================================

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
    time=[0.0],
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
    # Imports
    import matplotlib.pyplot as plt
    from matplotlib import colors as mpl_colors
    ####
    import mcdc.transport.kernel as kernel
    import mcdc.transport.geometry as geometry

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
