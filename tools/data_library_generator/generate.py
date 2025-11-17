import ACEtk
import h5py
import numpy as np
import os

####

import util
from util import print_error, print_note


# Directories
output_dir = os.getenv("MCDC_LIB")
ace_dir = os.getenv("MCDC_ACELIB")

# Create output directory if needed
os.makedirs(output_dir, exist_ok=True)
print(f'\nACE directory: {ace_dir}')
print(f'Output directory: {output_dir}\n')

distribution_count = {}

# Loop over all files
for ace_name in os.listdir(ace_dir):
    # Load ACE tables
    ace_table = ACEtk.ContinuousEnergyTable.from_file(f"{ace_dir}/{ace_name}")

    # Decode ACE name to MC/DC name
    Z, A, S, T = util.decode_ace_name(ace_table.header.zaid)
    symbol = util.Z_TO_SYMBOL[Z]
    nuclide_name = f"{symbol}{A}"
    mcdc_name = f"{nuclide_name}-{S}-{T}.h5"

    # Create MC/DC file
    print("\n"+"="*80+"\n")
    print(f'Create {mcdc_name} from {ace_name}\n')
    file = h5py.File(f"{output_dir}/{mcdc_name}", "w")

    # ==================================================================================
    # Basic properties
    # ==================================================================================

    # ACE data source description
    file.create_dataset("data_source/title", data=ace_table.header.title)
    file.create_dataset("data_source/version", data=ace_table.header.version)
    file.create_dataset("data_source/date", data=ace_table.header.date)
    if "comments" in dir(ace_table.header):
        file.create_dataset("data_source/comments", data=ace_table.header.comments)

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
    capture_group = reactions.create_group("capture")
    inelastic_group = reactions.create_group("inelastic_scattering")
    if fissionable:
        fission_group = reactions.create_group("fission")

    # Group MTs
    elastic_MTs = [2]
    capture_MTs = []
    inelastic_MTs = []
    fission_MTs = [18]
    redundant_MTs = [1, 3, 4, 10, 19, 20, 21, 38]
    fission_components = [19, 20, 21, 38]

    # Add MTs to capture and inelastic groups
    for i in range(N_reaction):
        idx = i + 1
        MT = rx_block.MT(idx)

        if MT in redundant_MTs + elastic_MTs + fission_MTs or MT > 117:
            continue

        nu = nu_block.multiplicity(idx)
        if nu == 0:
            capture_MTs.append(MT)
        elif nu > 0:
            inelastic_MTs.append(MT)
        else:
            print_error(f"Negative multiplicity for MT={MT}")

    # Report MT groups
    print(f"  Reaction group MTs")
    print(f"    - Elastic scattering MTs: {elastic_MTs}")
    print(f"    - Capture MTs: {capture_MTs}")
    print(f"    - Inelastic scattering MTs: {inelastic_MTs}")
    if fissionable:
        print(f"    - Fission MTs: {fission_MTs}")

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
    xs_energy = np.array(xs_energy) * 1e6  # MeV to eV
    xs_energy = reactions.create_dataset("xs_energy_grid", data=xs_energy)
    xs_energy.attrs["unit"] = "eV"

    # Elastic
    xs = elastic_group.create_dataset("MT-002/xs", data=xs_elastic)
    xs.attrs["energy_offset"] = 0
    xs.attrs["unit"] = "barns"

    # Capture, inelastic, and fission
    for i in range(N_reaction):
        idx = i + 1
        MT = rx_block.MT(idx)

        if MT in redundant_MTs or MT > 117:
            continue

        if MT in capture_MTs:
            group = capture_group
        elif MT in inelastic_MTs:
            group = inelastic_group
        elif fissionable and MT in fission_MTs:
            group = fission_group

        xs = group.create_dataset(f"MT-{MT:03}/xs", data=cross_sections(idx))
        xs.attrs["energy_offset"] = energy_offsets(idx) - 1
        xs.attrs["unit"] = "barns"

    # Fissionable, but provided by components
    if fissionable and not rx_block.has_MT(18):
        xs_fission = np.zeros_like(xs_energy)

        for MT in fission_components:
            idx = rx_block.index(MT)
            xs_fission[energy_offsets(idx) - 1:] += cross_sections(idx)[:]

        xs = fission_group.create_dataset("MT-018/xs", data=xs_fission)
        xs.attrs["energy_offset"] = 0
        xs.attrs["unit"] = "barns"
        

    # ==================================================================================
    # Reference frames and inelastic multiplicities
    # ==================================================================================

    # Elastic
    elastic_group.create_dataset("MT-002/reference_frame", data="LAB")

    # Capture, inelastic, and fission
    for i in range(N_reaction):
        idx = i + 1
        MT = rx_block.MT(idx)

        if MT in redundant_MTs or MT > 117:
            continue

        if MT in capture_MTs:
            group = capture_group
        elif MT in inelastic_MTs:
            group = inelastic_group
        elif fissionable and MT in fission_MTs:
            group = fission_group

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
    
    # Fissionable, but provided by components
    if fissionable and not rx_block.has_MT(18):
        xs_fission = np.zeros_like(xs_energy)

        reference_frames = []
        for MT in fission_components:
            idx = rx_block.index(MT)

            # Reference frame
            reference_frame = nu_block.reference_frame(idx)
            reference_frames.append(reference_frame)

        if len(set(reference_frames)) != 1:
            print_error("Fission component reference frames are identical")

        if reference_frames[0] == ACEtk.ReferenceFrame.Laboratory:
            reference_frame = 'LAB'
        elif reference_frames[0] == ACEtk.ReferenceFrame.CentreOfMass:
            reference_frame = 'COM'
        else:
            print_error(f"Unknown reaction reference frame type for MT={MT}")
        fission_group.create_dataset(f"MT-018/reference_frame", data=reference_frame)

    # ==================================================================================
    # Scattering angular distributions
    # ==================================================================================

    angle_block = ace_table.angular_distribution_block
    energy_correlated_MTs = []
   
    # Elastic scattering
    angle_group = elastic_group.create_group('MT-002/scattering_cosine')
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
        if data.__class__ in distribution_count.keys():
            distribution_count[data.__class__] += 1
        else:
            distribution_count[data.__class__] = 1

    # ==================================================================================
    # Fission multiplicities and delayed neutron precursor fractions and decay rates
    # ==================================================================================
    
    if fissionable:
        fission_reaction = fission_group['MT-018']
        prompt_block = ace_table.fission_multiplicity_block
        delayed_block = ace_table.delayed_fission_multiplicity_block
        dnp_block = ace_table.delayed_neutron_precursor_block

        # Prompt multiplicity
        data = prompt_block.multiplicity
        h5_group = fission_reaction.create_group("prompt_multiplicity")
        util.load_fission_multiplicity(data, h5_group)

        # Delayed multiplicity
        if delayed_block is not None:
            data = delayed_block.multiplicity
            h5_group = fission_reaction.create_group("delayed_multiplicity")
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
                    print("[ERROR] Non-constant delayed neutron precursor fraction")
                    exit()

                fractions[i] = data.probabilities[0]
                decay_rates[i] = data.decay_constant

            precursors = fission_reaction.create_group("delayed_neutron_precursors")
            precursors.create_dataset('fractions', data=fractions)
            decay_rates = precursors.create_dataset('decay_rates', data=decay_rates)
            decay_rates.attrs['unit'] = "/s"

    # ==================================================================================
    # Fission neuron angular and energy distributions
    # ==================================================================================
    
    # Fission angular emission: Isotropic (and energy correlated?)
    if fissionable:
        anisotropic = False

        # Report if anisotropic
        if rx_block.has_MT(18):
            idx = rx_block.index(18)
            if not angle_block.is_fully_isotropic(idx):
                anisotropic = True
        else:
            isotropic = []
            for MT in fission_components:
                idx = rx_block.index(MT)
                isotropic.append(angle_block.is_fully_isotropic(idx))
            if not all(isotropic):
                print_error('Anisotropic fission neutron')

        if anisotropic:
            idx = rx_block.index(18)

            # Energy-correlated?
            if isinstance(
                angle_block.angular_distribution_data(idx),
                ACEtk.continuous.DistributionGivenElsewhere
            ):
                angle_group.attrs['type'] = 'energy-correlated'
                energy_correlated_MTs.append(18)

            else:
                # Check distribution support: all tabulated
                data = angle_block.angular_distribution_data(idx)
                NE = data.number_incident_energies
                for i in range(NE):
                    idx = i + 1
                    if (
                        data.distribution_type(idx) != ACEtk.AngularDistributionType.Tabulated
                    ):
                        print_error("Anisotropic fission angular distribution is not all-tabulated")

                for i, distribution in enumerate(data.distributions):
                    pdf = distribution.pdf[:]
                    if len(pdf) != 2 or pdf[0] != 0.5 or pdf[1] != 0.5:
                        print_error("Anisotropic fission neutron")

                print_note("Tabulated isotropic fission neutron distribution")

    '''
    # Prompt neutron
    block = ace_table.energy_distribution_block
    N_reaction = block.number_reactions
    for i in range(N_reaction):
        idx = i + 1
        if MT_map(idx) == 18:
            distribution = block.energy_distribution_data(idx)
            break
        print(idx, MT_map(idx), distribution.type)
    exit()

    # Delayed neutrons
    block = ace_table.delayed_neutron_energy_distribution_block
    if block is not None:
        N_DNP = block.number_reactions
        for i in range(N_DNP):
            idx = i + 1
            distribution = block.energy_distribution_data(idx)
            if distribution.type != ACEtk.EnergyDistributionType.TabulatedEnergy:
                print("[ERROR] Non tabulated delayed neutron energy PDF")
                exit()
            print(distribution.incident_energies[:])


        for i in range(N_DNP):
            idx = 1 + 1
            data = (
                ace_table.delayed_neutron_precursor_block.precursor_group_data(idx)
            )
            
            if (
                not data.number_interpolation_regions == 0 
                or not len(data.probabilities[:]) == 2
                or not data.probabilities[0] == data.probabilities[1]
            ):
                print("[ERROR] Non-constant delayed neutron precursor fraction")
                exit()

            fractions[i] = data.probabilities[0]
            decay_rates[i] = data.decay_constant

        precursors = fission.create_group("delayed_neutron_precursors")
        precursors.create_dataset('fractions', data=fractions)
        decay_rates = precursors.create_dataset('decay_rates', data=decay_rates)
        decay_rates.attrs['unit'] = "/s"
    '''

    
    # ==================================================================================
    # Finalize
    # ==================================================================================

    file.close()

print("")

for key in distribution_count.keys():
    print(key, distribution_count[key])
