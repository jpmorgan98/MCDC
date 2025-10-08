import numpy as np

from numpy import float64, int64
from numpy.typing import NDArray
from types import NoneType
from typing import Annotated, Iterable

####

import mcdc.object_.distribution as distribution

from mcdc.object_.base import ObjectNonSingleton
from mcdc.constant import PARTICLE_NEUTRON
from mcdc.object_.distribution import DistributionPDF, DistributionPMF
from mcdc.object_.simulation import simulation


def decode_particle_type(type_):
    if type_ == PARTICLE_NEUTRON:
        return "Neutron"


# ======================================================================================
# Source
# ======================================================================================


class Source(ObjectNonSingleton):
    # Annotations for Numba mode
    label: str = 'source'
    #
    name: str
    point_source: bool
    box_source: bool
    point: Annotated[NDArray[float64], (3,)]
    x: Annotated[NDArray[float64], (2,)]
    y: Annotated[NDArray[float64], (2,)]
    z: Annotated[NDArray[float64], (2,)]
    isotropic_source: bool
    mono_direction: bool
    white_direction: bool
    direction: Annotated[NDArray[float64], (3,)]
    mono_energetic: bool
    energy_group: int
    energy: float
    energy_group_pmf: DistributionPMF
    energy_pdf: DistributionPDF
    discrete_time: bool
    time: float
    time_range: Annotated[NDArray[float64], (2,)]
    particle_type: int
    probability: float

    def __init__(self,
            name: str = "",
            position: Iterable[float] | NoneType = None,
            x: Iterable[float] | NoneType = None,
            y: Iterable[float] | NoneType = None,
            z: Iterable[float] | NoneType = None,
            #
            direction: Iterable[float] | NoneType = None,
            white_direction: Iterable[float] | NoneType = None,
            isotropic: bool | NoneType = None,
            #
            energy: float | NDArray[float64] | NoneType = None,
            energy_group: int | NDArray[int64] | NoneType = None,
            #
            time: float | Iterable[float] = 0.0,
            probability: float = 1.0
        ):
        
        super().__init__()

        # Set name
        if name != "":
            self.name = name
        else:
            self.name = f"{self.label}_{self.numba_ID}"

        # ==============================================================================
        # Default attributes
        #   Point source at origin, isotropic, mono-energetic at 1 MeV or at group 0, 
        #   time = 0, neutron
        # ==============================================================================

        # Position
        self.point_source = True
        self.box_source = False
        self.point = np.zeros(3)
        self.x = np.array([0.0, 0.0])
        self.y = np.array([0.0, 0.0])
        self.z = np.array([0.0, 0.0])

        # Direction
        self.isotropic_source = True
        self.mono_direction = False
        self.white_direction = False
        self.direction = np.array([0.0, 0.0, 0.0])

        # Energy
        self.mono_energetic = True
        self.energy_group = 0
        self.energy = 1.0E6
        self.energy_group_pmf = DistributionPMF(np.array([0.0]), np.array([1.0]))
        self.energy_pdf = DistributionPDF(np.array([1.E6 - 1.0, 1.E6 + 1.0]), np.array([1.0, 1.0]))

        # Time
        self.discrete_time = True
        self.time = 0.0
        self.time_range = np.array([0.0, 0.0])

        # Particle type
        self.particle_type = PARTICLE_NEUTRON

        # Probability
        self.probability = probability

        # ==============================================================================
        # Assignment
        # ==============================================================================
        
        # Position
        if position is not None:
            self.point = np.array(position)
        else:
            self.point_source = False
            self.box_source = True
            if x is not None:
                self.x = np.array(x)
            if y is not None:
                self.y = np.array(y)
            if z is not None:
                self.z = np.array(z)

        # Direction
        if isotropic is not None:
            pass
        elif direction is not None:
            self.isotropic_source = False
            self.mono_direction = True
            self.direction = np.array(direction)
        elif white_direction is not None:
            self.isotropic_source = False
            self.white_direction = False
            self.direction = np.array(white_direction)

        # Energy
        if energy_group is not None:
            if type(energy_group) == int:
                self.energy_group = energy_group
            else:
                self.mono_energetic = False
                self.energy_group_pmf = np.array(energy_group)
        elif energy is not None:
            if type(energy) == float:
                self.energy = energy
            else:
                self.mono_energetic = False
                self.energy_pdf = np.array(energy)

        # Time
        if type(time) == float:
            self.time = time
        else:
            self.discrete_time = False
            self.time_range = np.array(time)
   
        # ==============================================================================
        # Normalize probability
        # ==============================================================================

        norm = 0.0
        for source in simulation.sources:
            norm += source.probability
        for source in simulation.sources:
            source.probability /= norm

    def __repr__(self):
        text = "\n"
        text += f"Source\n"
        text += f"  - ID: {self.ID}\n"
        text += f"  - Name: {self.name}\n"
        text += f"  - Particle: {decode_particle_type(self.particle_type)}\n"
        text += f"  - Probability: {self.probability * 100}%\n"
        if self.point_source:
            text += f"  - Position [x, y, z]: {self.point} cm\n"
        else:
            text += f"  - Position\n"
            text += f"    - x: {self.x} cm\n"
            text += f"    - y: {self.y} cm\n"
            text += f"    - z: {self.z} cm\n"
        if self.isotropic_source:
            text += f"  - Direction: Isotropic\n"
        elif self.mono_direction:
            text += f"  - Direction [ux, uy, yz]: {self.direction}\n"
        elif self.white_direction:
            text += f"  - Isotropic halfspace: {self.direction}\n"
        if simulation.materials[0].label == 'multigroup_material':
            if self.mono_energetic:
                text += f"  - Energy group: {self.energy_group} \n"
            else:
                text += f"  - Energy group: {distribution.decode_type(self.energy_group_pmf)} [ID: {self.energy_group_pmf.ID}]\n"
        else:
            if self.mono_energetic:
                text += f"  - Energy: {self.energy} eV\n"
            else:
                text += f"  - Energy: {distribution.decode_type(self.energy_pdf)} [ID: {self.energy_pdf.ID}]\n"
        if self.discrete_time:
            text += f"  - Time: {self.time} s\n"
        else:
            text += f"  - Time: {self.time_range} s\n"

        return text
