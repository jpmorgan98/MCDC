from dataclasses import dataclass
from numpy import uint

####

from mcdc.constant import PARTICLE_NEUTRON
from mcdc.object_.base import ObjectBase


@dataclass
class ParticleData(ObjectBase):
    label: str = "particle_data"
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    t: float = 0.0
    ux: float = 0.0
    uy: float = 0.0
    uz: float = 0.0
    g: int = -1
    E: float = 0.0
    w: float = 0.0
    particle_type: int = PARTICLE_NEUTRON
    rng_seed: uint = uint(1)


@dataclass
class Particle(ParticleData):
    label: str = "particle"
    cell_ID: int = -1
    material_ID: int = -1
    surface_ID: int = -1
    alive: bool = False
    fresh: bool = False
    event: int = -1
