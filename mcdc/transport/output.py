import h5py
import importlib.metadata
import numpy as np

####

import mcdc.mcdc_get as mcdc_get
import mcdc.print_ as print_module

from mcdc.constant import (
    MESH_UNIFORM,
    MESH_STRUCTURED,
    SCORE_FLUX,
    SCORE_DENSITY,
    SCORE_COLLISION,
    SCORE_FISSION,
    SCORE_NET_CURRENT,
    TALLY_LITERALS,
)


# ======================================================================================
# Main output
# ======================================================================================

def generate_output(mcdc, data):
    settings = mcdc['settings']

    if not mcdc["mpi_master"]:
        return

    # Header
    if settings["use_progress_bar"]:
        print_module.print_msg("")
    print_module.print_msg(" Generating output HDF5 files...")

    # Create the file
    file = h5py.File(settings["output_name"] + ".h5", "w")

    # Version
    file["version"] = importlib.metadata.version("mcdc")

    # Input deck
    if settings["save_input_deck"]:
        input_group = file.create_group("input_deck")
        # cardlist_to_h5group(simulation.nuclides, input_group, "nuclide")
        # cardlist_to_h5group(simulation.materials, input_group, "material")
        # cardlist_to_h5group(input_deck.surfaces, input_group, "surface")
        # cardlist_to_h5group(input_deck.cells, input_group, "cell")
        # cardlist_to_h5group(input_deck.universes, input_group, "universe")
        # cardlist_to_h5group(input_deck.lattices, input_group, "lattice")
        # cardlist_to_h5group(input_deck.sources, input_group, "source")
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
        #dict_to_h5group(
        #    input_deck.technique, input_group.create_group("technique")
        #)

    # No need to output tally if time census-based tally is used
    if mcdc["settings"]["use_census_based_tally"]:
        return

    # Tallies
    create_tally_dataset(file, mcdc, data)

    # Eigenvalues
    if mcdc["settings"]["eigenvalue_mode"]:
        if mcdc["technique"]["iQMC"]:
            file.create_dataset("k_eff", data=mcdc["k_eff"])
            if mcdc["technique"]["iqmc"]["mode"] == "batched":
                N_cycle = mcdc["setting"]["N_cycle"]
                file.create_dataset("k_cycle", data=mcdc["k_cycle"][:N_cycle])
                file.create_dataset("k_mean", data=mcdc["k_avg_running"])
                file.create_dataset("k_sdev", data=mcdc["k_sdv_running"])
        else:
            N_cycle = mcdc["settings"]["N_cycle"]
            file.create_dataset("k_cycle", data=mcdc["k_cycle"][:N_cycle])
            file.create_dataset("k_mean", data=mcdc["k_avg_running"])
            file.create_dataset("k_sdev", data=mcdc["k_sdv_running"])
            file.create_dataset("global_tally/neutron/mean", data=mcdc["n_avg"])
            file.create_dataset("global_tally/neutron/sdev", data=mcdc["n_sdv"])
            file.create_dataset("global_tally/neutron/max", data=mcdc["n_max"])
            file.create_dataset("global_tally/precursor/mean", data=mcdc["C_avg"])
            file.create_dataset("global_tally/precursor/sdev", data=mcdc["C_sdv"])
            file.create_dataset("global_tally/precursor/max", data=mcdc["C_max"])
            if mcdc["settings"]["use_gyration_radius"]:
                file.create_dataset(
                    "gyration_radius", data=mcdc["gyration_radius"][:N_cycle]
                )

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
                file.create_dataset("particles", data=neutrons[:])
                file.create_dataset("particles_size", data=len(neutrons[:]))


    # Close the file
    file.close()


# ======================================================================================
# Runtimes
# ======================================================================================

def create_runtime_datasets(mcdc):
    import h5py
    import mcdc.config as config

    if not mcdc["mpi_master"]:
        return

    base_name = mcdc["settings"]["output_name"]

    print('HERE')
    main_output = h5py.File(f"{base_name}.h5", "a")
    create_runtime_dataset(main_output, mcdc)
    main_output.close()

    if config.args.runtime_output:
        print('THERE')
        runtime_output = h5py.File(f"{base_name}.h5", "w")
        create_runtime_dataset(runtime_output, mcdc)
        runtime_output.close()


def create_runtime_dataset(file, mcdc):
    for name in [
        "total",
        "preparation",
        "simulation",
        "output",
        "bank_management",
    ]:
        file.create_dataset(f"runtime/{name}", data=np.array([mcdc["runtime_" + name]]))


# ======================================================================================
# Tally recombination
# ======================================================================================

def replace_dataset(file_, field, data):
    if field in file_:
        del file_[field]
    file_.create_dataset(field, data=data)


def recombine_tallies(file_name="output.h5"):
    import h5py

    if MPI.COMM_WORLD.Get_rank() > 0:
        return

    # Load main output file and read input params
    with h5py.File(file_name, "r") as f:
        output_name = str(f["input_deck/settings/output_name"][()])[2:-1]
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


def generate_census_based_tally(mcdc, data):
    idx_batch = mcdc["idx_batch"]
    idx_census = mcdc["idx_census"]
    base_name = mcdc['settings']['output_name']

    # Create or get the file
    file_name = f"{base_name}-batch_{idx_batch}-census_{idx_census}.h5"
    file = h5py.File(file_name, 'w')
    create_tally_dataset(file, mcdc, data)
    file.close()


def create_tally_dataset(file, mcdc, data):
    # Loop over all tally types
    for tally_type in TALLY_LITERALS:
        # Loop over tallies
        for i in range(mcdc[f'N_{tally_type}_tally']):
            tally = mcdc[f'{tally_type}_tallies'][i]
            tally_name = tally['name']

            # Filter grids
            file.create_dataset(f"tallies/{tally_name}/grid/mu", data=mcdc_get.tally.mu_all(tally, data))
            file.create_dataset(f"tallies/{tally_name}/grid/azi", data=mcdc_get.tally.azi_all(tally, data))
            file.create_dataset(f"tallies/{tally_name}/grid/energy", data=mcdc_get.tally.energy_all(tally, data))
            file.create_dataset(f"tallies/{tally_name}/grid/time", data=mcdc_get.tally.time_all(tally, data))

            # Mesh grid
            if tally_type == 'mesh':
                mesh_type = tally['mesh_type']
                mesh_ID = tally['mesh_ID']
                if mesh_type == MESH_UNIFORM:
                    mesh = mcdc['uniform_meshes'][mesh_ID]
                    x = np.linspace(mesh['x0'], mesh['x0'] + mesh['dx'], mesh['Nx'] + 1)
                    y = np.linspace(mesh['y0'], mesh['y0'] + mesh['dy'], mesh['Ny'] + 1)
                    z = np.linspace(mesh['z0'], mesh['z0'] + mesh['dz'], mesh['Nz'] + 1)
                elif mesh_type == MESH_STRUCTURED:
                    mesh = mcdc['structured_meshes'][mesh_ID]
                    x = mcdc_get.structured_mesh.x_all(mesh, data)
                    y = mcdc_get.structured_mesh.y_all(mesh, data)
                    z = mcdc_get.structured_mesh.z_all(mesh, data)
                file.create_dataset(f"tallies/{tally_name}/grid/x", data=x)
                file.create_dataset(f"tallies/{tally_name}/grid/y", data=y)
                file.create_dataset(f"tallies/{tally_name}/grid/z", data=z)

            # Get and reshape tally
            N_bin = tally['bin_length']
            start_mean = tally['bin_sum_offset']
            start_sdev = tally['bin_sum_square_offset']
            mean = data[start_mean:start_mean+N_bin]
            sdev = data[start_sdev:start_sdev+N_bin]
            shape = tuple([int(x) for x in mcdc_get.tally.bin_shape_all(tally, data)])
            mean = mean.reshape(shape)
            sdev = sdev.reshape(shape)

            # Roll tally so that score is in the front
            roll_reference = 4
            if tally_type == 'mesh':
                roll_reference = 7
            mean = np.rollaxis(mean, roll_reference, 0)
            sdev = np.rollaxis(sdev, roll_reference, 0)

            # Iterate over scores
            for i in range(tally['scores_length']):
                score_type = mcdc_get.tally.scores(i, tally, data)
                score_mean = np.squeeze(mean[i])
                score_sdev = np.squeeze(sdev[i])
                if score_type == SCORE_FLUX:
                    score_name = "flux"
                elif score_type == SCORE_DENSITY:
                    score_name = "density"
                elif score_type == SCORE_COLLISION:
                    score_name = "total"
                elif score_type == SCORE_FISSION:
                    score_name = "fission"
                elif score_type == SCORE_NET_CURRENT:
                    score_name = "net-current"
                group_name = f"tallies/{tally_name}/{score_name}/"
                file.create_dataset(group_name + "mean", data=score_mean)
                file.create_dataset(group_name + "sdev", data=score_sdev)
