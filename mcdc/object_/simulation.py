from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mcdc.object_.cell import Cell, Region
    from mcdc.object_.material import MaterialBase
    from mcdc.object_.surface import Surface
    from mcdc.object_.tally import TallyBase

####

from mcdc.object_.base import ObjectSingleton
from mcdc.object_.data import DataBase
from mcdc.object_.distribution import DistributionBase
from mcdc.object_.mesh import MeshBase
from mcdc.object_.nuclide import Nuclide
from mcdc.object_.reaction import ReactionBase
from mcdc.object_.universe import Universe, Lattice
from mcdc.object_.settings import Settings


# ======================================================================================
# Simulation
# ======================================================================================


class Simulation(ObjectSingleton):
    label: str = 'simulation'

    # Annotations for Numba mode
    data: list[DataBase]
    distributions: list[DistributionBase]
    materials: list[MaterialBase]
    nuclides: list[Nuclide]
    reactions: list[ReactionBase]
    cells: list[Cell]
    lattices: list[Lattice]
    regions: list[Region]
    surfaces: list[Surface]
    universes: list[Universe]
    meshes: list[MeshBase]
    settings: Settings
    tallies: list[TallyBase]

    def __init__(self):
        super().__init__()

        # Physics
        self.data = []
        self.distributions = []
        self.materials = []
        self.nuclides = []
        self.reactions = []

        # Geometry
        self.cells = []
        self.lattices = []
        self.regions = []
        self.surfaces = []
        self.universes = [Universe("Root Universe", root=True)]

        # Others
        self.meshes = []
        self.settings = Settings()
        self.tallies = []

        self.non_numba += ['settings']

    def set_root_universe(self, cells=[]):
        self.universes[0].cells = cells


simulation = Simulation()
