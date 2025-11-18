import ACEtk
import argparse
import h5py
import numpy as np
import os

from tqdm import tqdm

####

import util
from util import print_error, print_note

parser = argparse.ArgumentParser(description="MC/DC data generator")
parser.add_argument("--rewrite", dest="rewrite", action="store_true", default=False)
parser.add_argument("--verbose", dest="verbose", action="store_true", default=False)
args, unargs = parser.parse_known_args()
rewrite = args.rewrite
verbose = args.verbose

# Directories
output_dir = os.getenv("MCDC_LIB")
ace_dir = os.getenv("MCDC_ACELIB")

if output_dir is None:
    print_error("Environment variable $MCDC_LIB is not set")
if ace_dir is None:
    print_error("Environment variable $MCDC_ACELIB is not set")

# Create output directory if needed
os.makedirs(output_dir, exist_ok=True)
print(f'\nACE directory: {ace_dir}')
print(f'Output directory: {output_dir}\n')

# Select the files
if rewrite:
    target_files = os.listdir(ace_dir)
else:
    target_files = []
    for file_name in os.listdir(ace_dir):
        # File header
        with open(f"{ace_dir}/{file_name}", 'r') as f:
            header = ACEtk.Header.from_string(f.readline())

        # Decode ACE name to MC/DC name
        Z, A, S, T = util.decode_ace_name(header.zaid)
        symbol = util.Z_TO_SYMBOL[Z]
        nuclide_name = f"{symbol}{A}"
        mcdc_name = f"{nuclide_name}-{S}-{T}.h5"
        
        if not os.path.exists(f"{output_dir}/{mcdc_name}"):
            target_files.append(file_name)

# Loop over all files
pbar = tqdm(target_files, disable=verbose, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt}{postfix}")
for ace_name in pbar:
    # File header
    with open(f"{ace_dir}/{ace_name}", 'r') as f:
        header = ACEtk.Header.from_string(f.readline())

    # Decode ACE name to MC/DC name
    Z, A, S, T = util.decode_ace_name(header.zaid)
    symbol = util.Z_TO_SYMBOL[Z]
    nuclide_name = f"{symbol}{A}"
    mcdc_name = f"{nuclide_name}-{S}-{T}.h5"
    
    if not rewrite and os.path.exists(f"{output_dir}/{mcdc_name}"):
        continue

    # Create MC/DC file
    if verbose:
        print("\n"+"="*80+"\n")
        print(f'Create {mcdc_name} from {ace_name}\n')
    pbar.set_postfix_str(f"{mcdc_name[:-3]} from {ace_name}")
    file = h5py.File(f"{output_dir}/{mcdc_name}", "w")

    # ==================================================================================
    # Basic properties
    # ==================================================================================

    # Load ACE tables
    ace_table = ACEtk.ContinuousEnergyTable.from_file(f"{ace_dir}/{ace_name}")

    # ACE data source description
    header = ace_table.header
    file.attrs['source_title'] = header.title
    file.attrs['source_version'] = header.version
    file.attrs['source_date'] = header.date
    if "comments" in dir(header):
        file.attrs['source_comments'] = header.comments

    # Name and excitation level
    file.create_dataset("nuclide_name", data=nuclide_name)
    file.create_dataset("excitation_level", data=S)

    # Temperature
    temperature = file.create_dataset("temperature", data=T)
    temperature.attrs['unit'] = 'K'

    # Atomic weight ratio
    atomic_weight_ratio = ace_table.atomic_weight_ratio
    file.create_dataset("atomic_weight_ratio", data=atomic_weight_ratio)

    # Fissionable?
    fissionable = ace_table.fission_multiplicity_block is not None
    file.create_dataset("fissionable", data=fissionable)

    # ==================================================================================
    # Reaction groups
    # ==================================================================================

    reactions = file.create_group("neutron_reactions")
    
    # ACE blocks
    nu_block = ace_table.frame_and_multiplicity_block
    rx_block = ace_table.reaction_number_block
    N_reaction = nu_block.number_reactions

    if nu_block.number_reactions != rx_block.number_reactions:
        print_error('Non-equal reaction number in reaction and multiplicity blocks')

    # The groups
    elastic_group = reactions.create_group("elastic_scattering")
    capture_group = reactions.create_group("captures")
    inelastic_group = reactions.create_group("inelastic_scatterings")
    fission_group = reactions.create_group("fission")

    # MT groups
    elastic_MT = 2
    capture_MTs = []
    inelastic_MTs = []
    fission_MT = 18
    fission_components = [19, 20, 21, 38]
    redundant_MTs = [1, 3, 4, 10] + fission_components

    # Add MTs to capture and inelastic groups
    for i in range(N_reaction):
        idx = i + 1
        MT = rx_block.MT(idx)

        if MT in redundant_MTs + [elastic_MT , fission_MT] or MT > 117:
            continue

        nu = nu_block.multiplicity(idx)
        if nu == 0:
            capture_MTs.append(MT)
        elif nu > 0:
            inelastic_MTs.append(MT)
        else:
            print_error(f"Negative multiplicity for MT={MT}")

    # Report MT groups
    if verbose:
        print(f"  Reaction group MTs")
        print(f"    - Elastic scattering MT: {elastic_MT}")
        print(f"    - Capture MTs: {capture_MTs}")
        print(f"    - Inelastic scattering MTs: {inelastic_MTs}")
        if fissionable:
            print(f"    - Fission MT: {fission_MT}")

    # Delete empty groups
    if not fissionable:
        del file['neutron_reactions/fission']
    if len(inelastic_MTs) == 0:
        del file['neutron_reactions/inelastic_scatterings']

    # ==================================================================================
    # Cross-sections
    # ==================================================================================

    xs0_block = ace_table.principal_cross_section_block
    xs_block = ace_table.cross_section_block

    xs_energy = xs0_block.energies
    xs_elastic = xs0_block.elastic
    cross_sections = xs_block.cross_sections
    energy_offsets = xs_block.energy_index

    # Energy grid
    xs_energy = np.array(xs_energy)
    dataset = reactions.create_dataset("xs_energy_grid", data=xs_energy)
    dataset.attrs["unit"] = "MeV"

    # Elastic
    xs = elastic_group.create_dataset("xs", data=xs_elastic)
    xs.attrs["energy_offset"] = 0
    xs.attrs["unit"] = "barns"

    # Capture and inelastic
    for i in range(N_reaction):
        idx = i + 1
        MT = rx_block.MT(idx)

        if MT in redundant_MTs + [elastic_MT , fission_MT] or MT > 117:
            continue

        if MT in capture_MTs:
            group = capture_group
        elif MT in inelastic_MTs:
            group = inelastic_group

        xs = group.create_dataset(f"MT-{MT:03}/xs", data=cross_sections(idx))
        xs.attrs["energy_offset"] = energy_offsets(idx) - 1
        xs.attrs["unit"] = "barns"

    # Fission
    if fissionable:
        total_fission_given = rx_block.has_MT(fission_MT)

        # Should be either
        if rx_block.has_MT(fission_MT):
            for MT in fission_components:
                if rx_block.has_MT(MT):
                    print_error('Both total fission and its components are given')

        xs_fission = np.zeros_like(xs_energy)

        # Total XS is given?
        if total_fission_given:
            idx = rx_block.index(fission_MT)
            xs_fission[energy_offsets(idx) - 1:] = cross_sections(idx)[:]

        # Accumulate from components if total is not given
        else:
            actual_fission_components = []
            for MT in fission_components:
                if rx_block.has_MT(MT):
                    actual_fission_components.append(MT)
                    idx = rx_block.index(MT)

                    xs_component = np.array(cross_sections(idx)[:])
                    xs = fission_group.create_dataset(f"MT-{MT:03}/xs", data=xs_component)
                    xs.attrs["energy_offset"] = energy_offsets(idx) - 1
                    xs.attrs["unit"] = "barns"

                    xs_fission[energy_offsets(idx) - 1:] += xs_component

        xs = fission_group.create_dataset("xs", data=xs_fission)
        xs.attrs["energy_offset"] = 0
        xs.attrs["unit"] = "barns"

    # ==================================================================================
    # Capture and inelastic reference frames and inelastic multiplicities
    # ==================================================================================

    for i in range(N_reaction):
        idx = i + 1
        MT = rx_block.MT(idx)

        if MT in redundant_MTs + [elastic_MT , fission_MT] or MT > 117:
            continue

        if MT in capture_MTs:
            group = capture_group
        elif MT in inelastic_MTs:
            group = inelastic_group

        # Reference frame
        reference_frame = nu_block.reference_frame(idx)
        if reference_frame == ACEtk.ReferenceFrame.Laboratory:
            reference_frame = 'LAB'
        elif reference_frame == ACEtk.ReferenceFrame.CentreOfMass:
            reference_frame = 'COM'
        else:
            print_error(f"Unknown reaction reference frame type for MT={MT}")
        group.create_dataset(f"MT-{MT:03}/reference_frame", data=reference_frame)

        # Inelastic multiplicity
        if MT in inelastic_MTs:
            nu = nu_block.multiplicity(idx)
            inelastic_group.create_dataset(f"MT-{MT:03}/multiplicity", data=nu)
    
    # Make sure that fission is in LAB frame
    if fissionable:
        if total_fission_given:
            idx = rx_block.index(fission_MT)
            if nu_block.reference_frame(idx) != ACEtk.ReferenceFrame.Laboratory:
                print_error(f"Fission reference frame is not LAB")
        else:
            for MT in actual_fission_components:
                idx = rx_block.index(MT)
                if nu_block.reference_frame(idx) != ACEtk.ReferenceFrame.Laboratory:
                    print_error(f"Fission reference frame is not LAB")

    # ==================================================================================
    # Scattering angular distributions
    # ==================================================================================

    angle_block = ace_table.angular_distribution_block
   
    # Elastic scattering
    angle_group = elastic_group.create_group('scattering_cosine')
    data = angle_block.angular_distribution_data(0)
    util.load_cosine_distribution(data, angle_group)

    # Inelastic scattering
    for MT in inelastic_MTs:
        idx = rx_block.index(MT)
        angle_group = inelastic_group.create_group(f'MT-{MT:03}/scattering_cosine')
        data = angle_block.angular_distribution_data(idx)
        util.load_cosine_distribution(data, angle_group)

    # ==================================================================================
    # Inelastic scattering energy distributions
    # ==================================================================================

    energy_block = ace_table.energy_distribution_block
    if angle_block.number_projectile_production_reactions != energy_block.number_reactions:
        print_error('Non-equal reaction number in angular and energy distribution blocks')

    for MT in inelastic_MTs:
        idx = rx_block.index(MT)
        data = energy_block.energy_distribution_data(idx)
        
        if isinstance(data, ACEtk.continuous.MultiDistributionData):
            N = data.number_distributions

            for i in range(N):
                probability = data.probability(i + 1)
                if (
                    not probability.number_interpolation_regions == 1
                    or any(np.array(probability.interpolants) != 1)
                ):
                    print_error("Unsupported multi-energy-distribution probabilities")

                energy_group = inelastic_group.create_group(f'MT-{MT:03}/energy_out-{i+1}')
                distribution = data.distribution(i+1)
                util.load_energy_distribution(distribution, energy_group)

                dataset = energy_group.create_dataset("probability_energy", data=probability.energies)
                dataset = energy_group.create_dataset("probability", data=probability.probabilities[:-1])
                dataset.attrs['unit'] = "MeV"
        else:
            energy_group = inelastic_group.create_group(f'MT-{MT:03}/energy_out-1')
            util.load_energy_distribution(data, energy_group)

            dataset = energy_group.create_dataset("probability_energy", data=np.array([0.0, 30.0]))
            dataset = energy_group.create_dataset("probability", data=np.array([1.0]))
            dataset.attrs['unit'] = "MeV"

    # Fissionable zone below
    if not fissionable:
        continue

    # ==================================================================================
    # Fission multiplicities and delayed neutron precursor fractions and decay rates
    # ==================================================================================
    
    prompt_block = ace_table.fission_multiplicity_block
    delayed_block = ace_table.delayed_fission_multiplicity_block
    dnp_block = ace_table.delayed_neutron_precursor_block

    # Prompt multiplicity
    data = prompt_block.multiplicity
    h5_group = fission_group.create_group("prompt_multiplicity")
    util.load_fission_multiplicity(data, h5_group)

    # Delayed multiplicity
    if delayed_block is not None:
        data = delayed_block.multiplicity
        h5_group = fission_group.create_group("delayed_multiplicity")
        util.load_fission_multiplicity(data, h5_group)

    # Delayed neutron precursor fractions and decay rates
    if dnp_block is not None:
        N_DNP = dnp_block.number_delayed_precursors
        fractions = np.zeros(N_DNP)
        decay_rates = np.zeros(N_DNP)

        for i in range(N_DNP):
            idx = 1 + 1
            data = dnp_block.precursor_group_data(idx)
            
            if (
                not data.number_interpolation_regions == 0 
                or not len(data.probabilities[:]) == 2
                or not data.probabilities[0] == data.probabilities[1]
            ):
                print_error("Non-constant delayed neutron precursor fraction")

            fractions[i] = data.probabilities[0]
            decay_rates[i] = data.decay_constant

        precursors = fission_group.create_group("delayed_neutron_precursors")
        precursors.create_dataset('fractions', data=fractions)
        decay_rates = precursors.create_dataset('decay_rates', data=decay_rates)
        decay_rates.attrs['unit'] = "/s"

    # ==================================================================================
    # Fission angular and energy distributions
    # ==================================================================================

    # Angular distribution
    if total_fission_given:
        idx = rx_block.index(fission_MT)
        angle_group = fission_group.create_group(f'emission_cosine')
        data = angle_block.angular_distribution_data(idx)
        util.load_cosine_distribution(data, angle_group)
    else:
        for MT in actual_fission_components:
            idx = rx_block.index(MT)
            angle_group = fission_group.create_group(f'MT-{MT:03}/emission_cosine')
            data = angle_block.angular_distribution_data(idx)
            util.load_cosine_distribution(data, angle_group)

    # Energy distribution
    if total_fission_given:
        idx = rx_block.index(fission_MT)
        data = energy_block.energy_distribution_data(idx)
        if isinstance(data, ACEtk.continuous.MultiDistributionData):
            print_error("Multi-distribution fission spectra")
        else:
            energy_group = fission_group.create_group(f'energy_out')
            util.load_energy_distribution(data, energy_group)
    else:
        for MT in actual_fission_components:
            idx = rx_block.index(MT)
            data = energy_block.energy_distribution_data(idx)
            if isinstance(data, ACEtk.continuous.MultiDistributionData):
                print_error("Multi-distribution fission spectra")
            else:
                energy_group = fission_group.create_group(f'MT-{MT:03}/energy_out')
                util.load_energy_distribution(data, energy_group)

    # Delayed neutron energy distribution
    delayed_spectrum_block = ace_table.delayed_neutron_energy_distribution_block
    if dnp_block is not None:
        N_DNP = dnp_block.number_delayed_precursors

        for i in range(N_DNP):
            idx = 1 + 1
            data = delayed_spectrum_block.energy_distribution_data(idx)

            if not isinstance(data, ACEtk.continuous.OutgoingEnergyDistributionData):
                print_error(f'Unsupported delayed fission neutron spectrum: {data}')
           
            energy_group = fission_group.create_group(f'delayed_neutron_precursors/energy_out-{i+1}')
            util.load_energy_distribution(data, energy_group)
    
    # ==================================================================================
    # Finalize
    # ==================================================================================

    file.close()

print("")
