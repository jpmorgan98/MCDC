from mcdc.object_.base import ObjectSingleton
from mcdc.object_.universe import Universe
from mcdc.object_.settings import Settings


# ======================================================================================
# Simulation
# ======================================================================================


class Simulation(ObjectSingleton):
    def __init__(self):
        label = "simulation"
        super().__init__(label)

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
