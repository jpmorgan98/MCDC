import ACEtk
import h5py
import numpy as np
import os

####

import util
from util import print_error, print_note

# ======================================================================================
# Setup
# ======================================================================================

# Directories
ace_dir = "/Users/ilhamvariansyah/nuclear_data/ace/Lib81"
output_dir = "/Users/ilhamvariansyah/nuclear_data/mcdc"

# ======================================================================================
# Generate MC/DC nuclear data files from ACE files
# ======================================================================================

# Create output directory if needed
os.makedirs(output_dir, exist_ok=True)
print(f'\nACE directory: {ace_dir}')
print(f'Output directory: {output_dir}\n')

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
    elastic_group = reactions.create_group("elastic")
    capture_group = reactions.create_group("capture")
    if fissionable:
        fission_group = reactions.create_group("fission")
    inelastic_group = reactions.create_group("inelastic")

    # Group MTs
    elastic_MTs = [2]
    fission_MTs = [18]
    capture_MTs = []
    inelastic_MTs = []
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

    # Report MT
    print(f"  Reaction group MTs")
    print(f"    - Elastic MTs: {elastic_MTs}")
    print(f"    - Capture MTs: {capture_MTs}")
    if fissionable:
        print(f"    - Fission MTs: {fission_MTs}")
    print(f"    - Inelastic MTs: {inelastic_MTs}")

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

    # Capture, fission, and inelastic
    for i in range(N_reaction):
        idx = i + 1
        MT = rx_block.MT(idx)

        if MT in redundant_MTs or MT > 117:
            continue

        if MT in capture_MTs:
            group = capture_group
        elif fissionable and MT in fission_MTs:
            group = fission_group
        elif MT in inelastic_MTs:
            group = inelastic_group

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

    # Capture, fission, and inelastic
    for i in range(N_reaction):
        idx = i + 1
        MT = rx_block.MT(idx)

        if MT in redundant_MTs or MT > 117:
            continue

        if MT in capture_MTs:
            group = capture_group
        elif fissionable and MT in fission_MTs:
            group = fission_group
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
    # Angular distributions
    # ==================================================================================

    angle_block = ace_table.angular_distribution_block
    energy_block = ace_table.energy_distribution_block
    
    if angle_block.number_projectile_production_reactions != energy_block.number_reactions:
        print_error('Non-equal reaction number in angular and energy distribution blocks')
   
    # Elastic: scattering cosine
    angle_group = elastic_group.create_group('MT-002/scattering_cosine')
    if angle_block.is_fully_isotropic(0):
        angle_group.attrs['type'] = 'isotropic'
    else:
        angle_group.attrs['type'] = 'multi-table'
        data = angle_block.angular_distribution_data(0)

        # Check distribution support: all tabulated
        NE = data.number_incident_energies
        for i in range(NE):
            idx = i + 1
            if (
                data.distribution_type(idx) != ACEtk.AngularDistributionType.Tabulated
            ):
                print_error("Elastic scattering angular distribution is not all-tabulated")

        # Incident energy
        energy = np.array(data.incident_energies) * 1E6 # MeV to eV
        energy = angle_group.create_dataset('energy', data=energy)
        energy.attrs['unit'] = 'eV'

        # Disstributions
        interpolation = np.zeros(NE, dtype=int)
        offset = np.zeros(NE, dtype=int)
        cosine = []
        pdf = []
        for i, distribution in enumerate(data.distributions):
            interpolation[i] = distribution.interpolation
            offset[i] = len(cosine)
            cosine.extend(distribution.cosines)
            pdf.extend(distribution.pdf)
        cosine = np.array(cosine)
        pdf = np.array(pdf)
        angle_group.create_dataset('interpolation', data=interpolation)
        angle_group.create_dataset('offset', data=offset)
        angle_group.create_dataset('value', data=cosine)
        angle_group.create_dataset('pdf', data=pdf)

    # Fission: Isotropic (mostly)
    if fissionable:
        anisotropic = False

        # Report if anisotropic
        if rx_block.has_MT(18):
            idx = rx_block.index(18)
            if not angle_block.is_fully_isotropic(idx):
                anisotropic = True
                print_note('Anisotropic fission neutron')
        else:
            isotropic = []
            for MT in fission_components:
                idx = rx_block.index(MT)
                isotropic.append(angle_block.is_fully_isotropic(idx))
            if not all(isotropic):
                print_error('Anisotropic fission neutron')
                # TODO

        if anisotropic:
            angle_group = fission_group.create_group('MT-018/emission_cosine')
            angle_group.attrs['type'] = 'multi-table'

            idx = rx_block.index(18)
            data = angle_block.angular_distribution_data(idx)

            # Check distribution support: all tabulated
            NE = data.number_incident_energies
            for i in range(NE):
                idx = i + 1
                if (
                    data.distribution_type(idx) != ACEtk.AngularDistributionType.Tabulated
                ):
                    print_error("Fission angular distribution is not all-tabulated")

            # Incident energy
            energy = np.array(data.incident_energies) * 1E6 # MeV to eV
            energy = angle_group.create_dataset('energy', data=energy)
            energy.attrs['unit'] = 'eV'

            # Disstributions
            interpolation = np.zeros(NE, dtype=int)
            offset = np.zeros(NE, dtype=int)
            cosine = []
            pdf = []
            for i, distribution in enumerate(data.distributions):
                interpolation[i] = distribution.interpolation
                offset[i] = len(cosine)
                cosine.extend(distribution.cosines)
                pdf.extend(distribution.pdf)

                print(data.incident_energies[:])
                print(distribution.cosines[:])
                print(distribution.pdf[:])
                input()
            cosine = np.array(cosine)
            pdf = np.array(pdf)
            angle_group.create_dataset('interpolation', data=interpolation)
            angle_group.create_dataset('offset', data=offset)
            angle_group.create_dataset('value', data=cosine)
            angle_group.create_dataset('pdf', data=pdf)

    # Inelastic
    for MT in inelastic_MTs:
        idx = rx_block.index(MT)
        angle_group = inelastic_group.create_group(f'MT-{MT:03}/emission_cosine')

        if angle_block.is_fully_isotropic(idx):
            angle_group.attrs['type'] = 'isotropic'

        elif isinstance(
            angle_block.angular_distribution_data(idx),
            ACEtk.continuous.DistributionGivenElsewhere
        ):
            angle_group.attrs['type'] = 'energy-correlated'

        else:
            angle_group.attrs['type'] = 'multi-table'
            data = angle_block.angular_distribution_data(idx)

            # Check distribution support: all tabulated
            NE = data.number_incident_energies
            for i in range(NE):
                idx = i + 1
                if (
                    data.distribution_type(idx) != ACEtk.AngularDistributionType.Tabulated
                ):
                    print_error("Inelastic reaction angular distribution is not all-tabulated")

            # Incident energy
            energy = np.array(data.incident_energies) * 1E6 # MeV to eV
            energy = angle_group.create_dataset('energy', data=energy)
            energy.attrs['unit'] = 'eV'

            # Disstributions
            interpolation = np.zeros(NE, dtype=int)
            offset = np.zeros(NE, dtype=int)
            cosine = []
            pdf = []
            for i, distribution in enumerate(data.distributions):
                interpolation[i] = distribution.interpolation
                offset[i] = len(cosine)
                cosine.extend(distribution.cosines)
                pdf.extend(distribution.pdf)
            cosine = np.array(cosine)
            pdf = np.array(pdf)
            angle_group.create_dataset('interpolation', data=interpolation)
            angle_group.create_dataset('offset', data=offset)
            angle_group.create_dataset('value', data=cosine)
            angle_group.create_dataset('pdf', data=pdf)

    # ==================================================================================
    # Energy distributions
    # ==================================================================================
    
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
