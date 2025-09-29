import numpy as np

####

from mcdc.constant import INF, MESH_STRUCTURED, MESH_UNIFORM
from mcdc.object_.base import ObjectPolymorphic
from mcdc.print_ import print_1d_array


# ======================================================================================
# Mesh base class
# ======================================================================================


class MeshBase(ObjectPolymorphic):
    def __init__(self, label, type_, name):
        super().__init__(label, type_)

        self.name = f"{label}_{self.numba_ID}"
        if name is not None:
            self.name = name

        self.N_bin = 0

    def __repr__(self):
        text = "\n"
        text += f"{decode_type(self.type)}\n"
        text += f"  - ID: {self.ID}\n"
        text += f"  - Name: {self.name}\n"
        text += f"  - # of bins: {self.N_bin}\n"
        return text


def decode_type(type_):
    if type_ == MESH_UNIFORM:
        return "Uniform mesh"
    elif type_ == MESH_STRUCTURED:
        return "Structured mesh"


# ======================================================================================
# Uniform mesh
# ======================================================================================


class MeshUniform(MeshBase):
    def __init__(self, name=None, x=None, y=None, z=None, t=None):
        label = "uniform_mesh"
        type_ = MESH_UNIFORM
        super().__init__(label, type_, name)

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
        self.t0 = 0.0
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
        if t is not None:
            self.t0 = t[0]
            self.dt = t[1]
            self.Nt = t[2]

        self.N_bin = self.Nx * self.Ny * self.Nz * self.Nt

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Grid specification\n"
        text += f"    - x0/dx/Nx: {self.x0}/{self.dx}/{self.Nx} [cm]\n"
        text += f"    - y0/dy/Ny: {self.y0}/{self.dy}/{self.Ny} [cm]\n"
        text += f"    - z0/dz/Nz: {self.z0}/{self.dz}/{self.Nz} [cm]\n"
        text += f"    - t0/dt/Nt: {self.t0}/{self.dt}/{self.Nt} [s]\n"
        return text


# ======================================================================================
# Structured mesh
# ======================================================================================


class MeshStructured(MeshBase):
    def __init__(self, name=None, x=None, y=None, z=None, t=None):
        label = "structured_mesh"
        type_ = MESH_STRUCTURED
        super().__init__(label, type_, name)

        # Default uniform grids
        self.x = np.array([-INF, INF])
        self.y = np.array([-INF, INF])
        self.z = np.array([-INF, INF])
        self.t = np.array([0.0, INF])

        # Set the grid
        if x is not None:
            self.x = x
        if y is not None:
            self.y = y
        if z is not None:
            self.z = z
        if t is not None:
            self.t = t

        self.Nx = len(self.x) - 1
        self.Ny = len(self.y) - 1
        self.Nz = len(self.z) - 1
        self.Nt = len(self.t) - 1

        self.N_bin = self.Nx * self.Ny * self.Nz * self.Nt

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Grid specification\n"
        text += f"    - x {print_1d_array(self.x)} cm\n"
        text += f"    - y {print_1d_array(self.y)} cm\n"
        text += f"    - z {print_1d_array(self.z)} cm\n"
        text += f"    - t {print_1d_array(self.t)} s\n"
        return text
