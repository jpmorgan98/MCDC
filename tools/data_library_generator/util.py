import ACEtk
import h5py
import numpy as np


def print_error(message):
    print(f"\n  [ERROR]: {message}")
    exit()

def print_note(message):
    print(f"\n  [NOTE]: {message}")

def decode_interpolation(code):
    if code not in INTERPOLATION_MAP.keys():
        print_error(f"Unsupported interpolation law: {code}") 
    return INTERPOLATION_MAP[code]


def decode_ace_name(name: str):
    """
    Decode an ACE file name into atomic number Z, mass number A, excitation state S,
    following the rule:
        ZAID = 1000*Z + A,                      (ground state),
        ZAID = 1000*Z + A + 300 + 100*S,        (excited, S >= 1),
    and temperature T.
    Returns (Z, A, S, T)
    """
    zaid, extension = name.split(".")

    zaid = int(zaid)
    Z = zaid // 1000
    remainder = zaid % 1000

    if remainder < 300:
        # ground state
        A = remainder
        S = 0
    else:
        # excited state
        offset = remainder - 300
        S = offset // 100
        A = offset % 100

    T = ACE_TEMPERATURE_LIB81[extension]

    return Z, A, S, T


def get_zaid(nuclide_name):
    nuclide_name = nuclide_name.strip().capitalize()

    # Find where the letters end and digits begin
    symbol = ""
    mass = 0
    for i, ch in enumerate(nuclide_name):
        if ch.isdigit():
            symbol = nuclide_name[:i]
            mass = int(nuclide_name[i:])
            break
    else:
        raise ValueError(f"No mass number found in '{nuclide_name}'")

    if symbol not in Z_MAP.keys():
        raise ValueError(f"Unknown element symbol '{symbol}'")

    Z = Z_MAP[symbol]
    A = mass
    return Z, A


def get_ace_name(Z, A, T, S=None):
    ID = Z * 1000 + A
    if S is not None:
        ID += 300 + S * 100
    extension = ACE_EXTENSION_LIB81[T]
    return f"{ID}{extension}"


def load_fission_multiplicity(data, h5_group: h5py.Group):
    # Polynomial
    if data.type == 1:
        # Convert coefficients from MeV-based to eV-based
        C = np.array(data.coefficients)
        for i in range(len(C)):
            C[i] /= (1e6) ** i

        h5_group.attrs['type'] = 'polynomial'
        h5_group.create_dataset("coefficient", data=C)

    # Tabulated
    elif data.type == 2:
        if not data.interpolation_data.is_linear_linear:
            print(f"[ERROR] Non linear-linear tabulated multiplicity is not supported")
            exit()
       
        energy = np.array(data.energies) * 1E6 # MeV to eV

        h5_group.attrs['type'] = 'table'
        h5_group.create_dataset("value", data=data.multiplicities)
        h5_group.create_dataset("energy", data=energy)

    ## Yield - Unsupported
    else:
        print(f"[ERROR] Unsupported multiplicity type: {data.type}")
        exit()


def load_cosine_distribution(data, h5_group: h5py.Group):
    if isinstance(
        data,
        ACEtk.continuous.FullyIsotropicDistribution
    ):
        h5_group.attrs['type'] = 'isotropic'

    elif isinstance(
        data,
        ACEtk.continuous.DistributionGivenElsewhere
    ):
        h5_group.attrs['type'] = 'energy-correlated'

    else:
        h5_group.attrs['type'] = 'tabulated'

        # Check distribution support: all tabulated
        NE = data.number_incident_energies
        for i in range(NE):
            idx = i + 1
            if (
                data.distribution_type(idx) != ACEtk.AngularDistributionType.Tabulated
            ):
                print_error("Angular distribution is not all-tabulated")

        # Incident energy
        energy = np.array(data.incident_energies) * 1E6 # MeV to eV
        energy = h5_group.create_dataset('energy', data=energy)
        energy.attrs['unit'] = 'eV'

        # Tabulated disstributions
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
        h5_group.create_dataset('interpolation', data=interpolation)
        h5_group.create_dataset('offset', data=offset)
        h5_group.create_dataset('value', data=cosine)
        h5_group.create_dataset('pdf', data=pdf)


def load_energy_distribution(data, h5_group: h5py.Group):
    if isinstance(data, ACEtk.continuous.LevelScatteringDistribution):
        h5_group.attrs['type'] = 'level-scattering'

        C1 = np.array(data.C1) * 1E6 # MeV to eV
        C1 = h5_group.create_dataset("C1", data=C1)
        C1.attrs['unit'] = 'eV'

        h5_group.create_dataset("C2", data=data.C2)

    elif isinstance(data, ACEtk.continuous.EvaporationSpectrum):
        h5_group.attrs['type'] = 'evaporation'

        # MeV to eV
        energy = np.array(data.energies) * 1E6 
        temperature = np.array(data.temperatures) * 1E6 
        restriction_energy = np.array(data.restriction_energy) * 1E6 

        dataset = h5_group.create_dataset('energy', data=energy)
        dataset.attrs['unit'] = 'eV'
        dataset = h5_group.create_dataset('temperature', data=temperature)
        dataset.attrs['unit'] = 'eV'
        dataset = h5_group.create_dataset('restriction_energy', data=restriction_energy)
        dataset.attrs['unit'] = 'eV'

    elif isinstance(data, ACEtk.continuous.KalbachMannDistributionData):
        h5_group.attrs['type'] = 'kalbach-mann'

        if not data.interpolation_data.is_linear_linear:
            print_error("Non-linearly-interpolated kalbach-mann is not supported")
        
        energy = np.array(data.incident_energies[:]) * 1E6 # MeV to eV
        for i in range(data.number_incident_energies):
            distribution = data.distribution(i + 1)
            distribution.pdf

        # Check distribution support: all kalbach-mann
        NE = data.number_incident_energies

        # Incident energy
        energy = np.array(data.incident_energies) * 1E6 # MeV to eV
        energy = h5_group.create_dataset('energy', data=energy)
        energy.attrs['unit'] = 'eV'

        # Tabulated disstributions
        offset = np.zeros(NE, dtype=int)
        precompound_factor = []
        angular_slope = []
        energy_out = []
        pdf = []
        for i, distribution in enumerate(data.distributions):
            offset[i] = len(pdf)
            precompound_factor.extend(distribution.precompound_fraction_values)
            angular_slope.extend(distribution.angular_distribution_slope_values)
            energy_out.extend(distribution.outgoing_energies)
            pdf.extend(distribution.pdf)

        precompound_factor = np.array(precompound_factor)
        angular_slope = np.array(angular_slope)
        energy_out = np.array(energy_out) * 1E6 # MeV to eV
        pdf = np.array(pdf)

        h5_group.create_dataset('offset', data=offset)
        h5_group.create_dataset('precompound_factor', data=precompound_factor)
        h5_group.create_dataset('angular_slope', data=angular_slope)
        h5_group.create_dataset('energy_out', data=energy_out)
        h5_group.create_dataset('pdf', data=pdf)

    elif isinstance(data, ACEtk.continuous.EnergyAngleDistributionData):
        h5_group.attrs['type'] = 'energy-angle-tabulated'

        if not data.interpolation_data.is_linear_linear:
            print_error("Non-linearly-interpolated correlated-energy-angle is not supported")
        
        energy = np.array(data.incident_energies[:]) * 1E6 # MeV to eV
        for i in range(data.number_incident_energies):
            distribution = data.distribution(i + 1)
            distribution.pdf

        # Check distribution support: all kalbach-mann
        NE = data.number_incident_energies

        # Incident energy
        energy = np.array(data.incident_energies) * 1E6 # MeV to eV
        energy = h5_group.create_dataset('energy', data=energy)
        energy.attrs['unit'] = 'eV'

        # Tabulated distributions
        offset = np.zeros(NE, dtype=int)
        pdf = []
        energy_out = []
        inner_offset = []
        inner_pdf = []
        cosine = []
        for i, distribution in enumerate(data.distributions):
            offset[i] = len(pdf)
            pdf.extend(distribution.pdf)
            energy_out.extend(distribution.outgoing_energies)

            for j, inner_distribution in enumerate(distribution.distributions):
                inner_offset.append(len(inner_pdf))
                cosine.extend(inner_distribution.cosines)
                inner_pdf.extend(inner_distribution.pdf)

        pdf = np.array(pdf)
        energy_out = np.array(energy_out) * 1E6 # MeV to eV
        
        inner_offset = np.array(inner_offset)
        inner_pdf = np.array(inner_pdf)
        cosine = np.array(cosine)

        h5_group.create_dataset('offset', data=offset)
        h5_group.create_dataset('pdf', data=pdf)
        h5_group.create_dataset('energy_out', data=energy_out)
        h5_group.create_dataset('inner_offset', data=inner_offset)
        h5_group.create_dataset('inner_pdf', data=inner_pdf)
        h5_group.create_dataset('cosine', data=cosine)

    else:
        print_error(f"Unsupported energy distribution: {data}")


# ======================================================================================
# Constants
# ======================================================================================

INTERPOLATION_MAP = {
    2: "linear-linear"
}

ACE_TEMPERATURE_LIB81 = {
    "10c": 293.6,
    "11c": 600.0,
    "12c": 900.0,
    "13c": 1200.0,
    "14c": 2500.0,
    "15c": 0.1,
    "16c": 233.15,
    "17c": 273.15,
}

SYMBOL_TO_Z = {
    "H": 1,
    "He": 2,
    "Li": 3,
    "Be": 4,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Ne": 10,
    "Na": 11,
    "Mg": 12,
    "Al": 13,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
    "Ar": 18,
    "K": 19,
    "Ca": 20,
    "Sc": 21,
    "Ti": 22,
    "V": 23,
    "Cr": 24,
    "Mn": 25,
    "Fe": 26,
    "Co": 27,
    "Ni": 28,
    "Cu": 29,
    "Zn": 30,
    "Ga": 31,
    "Ge": 32,
    "As": 33,
    "Se": 34,
    "Br": 35,
    "Kr": 36,
    "Rb": 37,
    "Sr": 38,
    "Y": 39,
    "Zr": 40,
    "Nb": 41,
    "Mo": 42,
    "Tc": 43,
    "Ru": 44,
    "Rh": 45,
    "Pd": 46,
    "Ag": 47,
    "Cd": 48,
    "In": 49,
    "Sn": 50,
    "Sb": 51,
    "Te": 52,
    "I": 53,
    "Xe": 54,
    "Cs": 55,
    "Ba": 56,
    "La": 57,
    "Ce": 58,
    "Pr": 59,
    "Nd": 60,
    "Pm": 61,
    "Sm": 62,
    "Eu": 63,
    "Gd": 64,
    "Tb": 65,
    "Dy": 66,
    "Ho": 67,
    "Er": 68,
    "Tm": 69,
    "Yb": 70,
    "Lu": 71,
    "Hf": 72,
    "Ta": 73,
    "W": 74,
    "Re": 75,
    "Os": 76,
    "Ir": 77,
    "Pt": 78,
    "Au": 79,
    "Hg": 80,
    "Tl": 81,
    "Pb": 82,
    "Bi": 83,
    "Po": 84,
    "At": 85,
    "Rn": 86,
    "Fr": 87,
    "Ra": 88,
    "Ac": 89,
    "Th": 90,
    "Pa": 91,
    "U": 92,
    "Np": 93,
    "Pu": 94,
    "Am": 95,
    "Cm": 96,
    "Bk": 97,
    "Cf": 98,
    "Es": 99,
    "Fm": 100,
}

Z_TO_SYMBOL = {value: key for key, value in SYMBOL_TO_Z.items()}
