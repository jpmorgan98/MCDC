import numpy as np

####

from mcdc.constant import INF
from mcdc.object_.base import ObjectNonSingleton
from mcdc.util import flatten

# ======================================================================================
# Universe
# ======================================================================================


class Universe(ObjectNonSingleton):
    def __init__(self, name="", cells=[], root=False):
        # Custom treatment for root universe
        label = "universe"
        if root:
            super().__init__(label, register=False)
            self.ID = 0
        else:
            super().__init__(label)

        self.name = f"{label}_{self.numba_ID}"
        if name is not None:
            self.name = name
        self.cells = cells

    def __repr__(self):
        text = "\n"
        text += f"Universe\n"
        if self.ID == 0:
            text += f"  - ID: {self.ID} (root)\n"
        else:
            text += f"  - ID: {self.ID}\n"
        text += f"  - Name: {self.name}\n"
        text += f"Cells: {[x.ID for x in self.cells]}"
        return text


# ======================================================================================
# Lattice
# ======================================================================================


class Lattice(ObjectNonSingleton):
    def __init__(self, name="", x=None, y=None, z=None, universes=None):
        label = "lattice"
        super().__init__(label)

        self.name = f"{label}_{self.numba_ID}"
        if name is not None:
            self.name = name

        # Default uniform grids
        self.x0 = -INF
        self.dx = 2 * INF
        self.Nx = 1
        self.y0 = -INF
        self.dy = 2 * INF
        self.Ny = 1
        self.z0 = -INF
        self.dz = 2 * INF
        self.Nz = 1
        self.t0 = 0.0  # Placeholder time grid is needed to use mesh indexing function
        self.dt = INF
        self.Nt = 1

        # Set the grid
        if x is not None:
            self.x0 = x[0]
            self.dx = x[1]
            self.Nx = x[2]
        if y is not None:
            self.y0 = y[0]
            self.dy = y[1]
            self.Ny = y[2]
        if z is not None:
            self.z0 = z[0]
            self.dz = z[1]
            self.Nz = z[2]

        # Set universe IDs
        get_ID = np.vectorize(lambda obj: obj.ID)
        universe_IDs = get_ID(universes)
        ax_expand = []
        if x is None:
            ax_expand.append(2)
        if y is None:
            ax_expand.append(1)
        if z is None:
            ax_expand.append(0)
        for ax in ax_expand:
            universe_IDs = np.expand_dims(universe_IDs, axis=ax)

        # Change indexing structure: [z(flip), y(flip), x] --> [x, y, z]
        universe_IDs = np.transpose(universe_IDs)
        universe_IDs = np.flip(universe_IDs, axis=1)
        universe_IDs = np.flip(universe_IDs, axis=2)

        # Set up the universes
        Nx = self.Nx
        Ny = self.Ny
        Nz = self.Nz
        self.universes = []
        for ix in range(Nx):
            self.universes.append([])
            for iy in range(Ny):
                self.universes[-1].append([])
                for iz in range(Nz):
                    ID = universe_IDs[ix, iy, iz]
                    self.universes[-1][-1].append(simulation.universes[ID])

    def __repr__(self):
        text = "\n"
        text += f"Lattice\n"
        text += f"  - ID: {self.ID}\n"
        text += f"  - Name: {self.name}\n"
        text += f"  - (x0, dx, Nx): ({self.x0}, {self.dx}, {self.Nx})\n"
        text += f"  - (y0, dy, Ny): ({self.y0}, {self.dy}, {self.Ny})\n"
        text += f"  - (z0, dz, Nz): ({self.z0}, {self.dz}, {self.Nz})\n"
        text += f"Universes: {set([x.ID for x in list(flatten(self.universes))])}"
        return text
