from dataclasses import dataclass

from numpy import float64, int64, uint64
from mcdc.object_.base import ObjectBase


@dataclass
class ParticleData(ObjectBase):
    x: float64
    y: float64
    z: float64
    t: float64
    ux: float64
    uy: float64
    uz: float64
    g: int64
    E: float64
    w: float64
    particle_type: int64
    rng_seed: uint64


@dataclass
class Particle(ParticleData):
    cell_ID: int64
    material_ID: int64
    surface_ID: int64
    alive: bool
    fresh: bool
    event: int64
