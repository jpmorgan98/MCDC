import numpy as np
import sympy

from mcdc.constant import (
    BOOL_AND,
    BOOL_OR,
    BOOL_NOT,
    INF,
    PARTICLE_NEUTRON,
    PI,
)

# Get the global variable container
import mcdc.global_ as global_


class InputCard:
    def __init__(self, tag):
        self.tag = tag

    def __str__(self):
        text = "%s card\n" % self.tag

        for name in [
            a
            for a in dir(self)
            if not a.startswith("__")
            and not callable(getattr(self, a))
            and a != "tag"
            and not a.startswith("_")
        ]:
            text += "  %s : %s\n" % (name, str(getattr(self, name)))
        return text


class SourceCard(InputCard):
    def __init__(self):
        InputCard.__init__(self, "Source")

        # Set card data
        self.ID = None
        self.box = False
        self.isotropic = True
        self.white = False
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.box_x = np.array([0.0, 0.0])
        self.box_y = np.array([0.0, 0.0])
        self.box_z = np.array([0.0, 0.0])
        self.ux = 0.0
        self.uy = 0.0
        self.uz = 0.0
        self.white_x = 0.0
        self.white_y = 0.0
        self.white_z = 0.0
        self.group = np.array([1.0])
        self.energy = np.array([[1e6 - 1.0, 1e6 + 1.0], [1.0, 1.0]])
        self.time = np.array([0.0, 0.0])
        self.particle_type = PARTICLE_NEUTRON
        self.prob = 1.0


# ======================================================================================
# Tally cards
# ======================================================================================


class TallyCard(InputCard):
    def __init__(self, type_):
        InputCard.__init__(self, type_)

        # Set card data
        self.ID = None
        self.scores = []
        self.N_bin = 0

        # Filters
        self.t = np.array([-INF, INF])
        self.mu = np.array([-1.0, 1.0])
        self.azi = np.array([-PI, PI])
        self.g = np.array([-INF, INF])


class MeshTallyCard(TallyCard):
    def __init__(self):
        TallyCard.__init__(self, "Mesh tally")

        # Set card data
        self.x = np.array([-INF, INF])
        self.y = np.array([-INF, INF])
        self.z = np.array([-INF, INF])
        self.N_bin = 1


class SurfaceTallyCard(TallyCard):
    def __init__(self, surface_ID):
        TallyCard.__init__(self, "Surface tally")

        # Set card data
        self.surface_ID = surface_ID
        self.N_bin = 1


class CellTallyCard(TallyCard):
    def __init__(self, cell_ID):
        TallyCard.__init__(self, "Cell tally")

        # Set card data
        self.cell_ID = cell_ID
        self.N_bin = 1


class CSTallyCard(TallyCard):
    def __init__(self):
        TallyCard.__init__(self, "CS tally")

        # Set card data
        self.x = np.array([-INF, INF])
        self.y = np.array([-INF, INF])
        self.z = np.array([-INF, INF])
        self.N_bin = 1
        self.N_cs_bins = 1
        self.cs_bin_size = np.array([1.0, 1.0])
