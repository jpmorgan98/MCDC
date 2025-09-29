import numpy as np

####

import mcdc.object_.mesh as mesh_module

from mcdc.constant import (
    INF,
    MESH_STRUCTURED,
    MESH_UNIFORM,
    PI,
    SCORE_FLUX,
    TALLY_CELL,
    TALLY_MESH,
    TALLY_SURFACE,
)
from mcdc.object_.base import ObjectPolymorphic
from mcdc.object_.simulation import simulation
from mcdc.print_ import print_1d_array


# ======================================================================================
# Tally base class
# ======================================================================================


class TallyBase(ObjectPolymorphic):
    def __init__(
        self, label, type_, name, scores, mu, azi, polar_reference, energy, time
    ):
        super().__init__(label, type_)

        self.name = f"{label}_{self.numba_ID}"
        if name is not None:
            self.name = name

        # Set scores
        self.scores = []
        for score in scores:
            if score == "flux":
                self.scores.append(SCORE_FLUX)

        # Phase-space filters
        self.mu = np.array([-1.0, 1.0])
        self.azi = np.array([-PI, PI])
        self.polar_reference = np.array([0.0, 0.0, 1.0])
        self.energy = np.array([-1.0, INF])
        self.time = np.array([0.0, INF])
        self.filter_direction = False
        self.filter_energy = False
        self.filter_time = False
        if mu is not None:
            self.mu = mu
            self.filter_direction = True
        if azi is not None:
            self.azi = azi
            self.filter_direction = True
        if polar_reference is not None:
            self.polar_reference /= polar_reference / np.linalg.norm(polar_reference)
        if energy is not None:
            if energy == "all_groups":
                G = simulation.materials[0].G
                self.energy = np.linspace(0, G, G + 1) - 0.5
            else:
                self.energy = energy
            self.filter_energy = True
        if time is not None:
            self.time = time
            self.filter_time = True

        # Tally bins (will be allocated by the subclass)
        self.bin = None
        self.bin_sum = None
        self.bin_sum_square = None

        # Strides (will be set by the subclass)
        self.stride_time = 0
        self.stride_energy = 0
        self.stride_azi = 0
        self.stride_mu = 0

    def _phasespace_filter_text(self):
        text = ""
        text += f"  - Scores: {[decode_score_type(x) for x in self.scores]}\n"
        text += f"  - Phase-space filters\n"
        if self.filter_time:
            text += f"    - Time {print_1d_array(self.time)} s\n"
        if self.filter_energy:
            text += f"    - Energy {print_1d_array(self.energy)} eV\n"
        if self.filter_direction:
            text += f"    - Direction\n"
            text += f"    -   Polar reference: {self.polar_reference}\n"
            text += f"    -   Polar cosine {print_1d_array(self.mu)}\n"
            text += f"    -   Azimuthal angle {print_1d_array(self.azi)}\n"
        return text

    def __repr__(self):
        text = "\n"
        text += f"{decode_type(self.type)}\n"
        text += f"  - ID: {self.ID}\n"
        text += f"  - Name: {self.name}\n"
        return text


def decode_type(type_):
    if type_ == TALLY_CELL:
        return "Cell tally"
    elif type_ == TALLY_SURFACE:
        return "Surface tally"
    elif type_ == TALLY_MESH:
        return "Mesh tally"


def decode_score_type(type_):
    if type_ == SCORE_FLUX:
        return "Flux"


# ======================================================================================
# Cell tally
# ======================================================================================


class TallyCell(TallyBase):
    def __init__(
        self,
        name=None,
        cell=None,
        scores=None,
        mu=None,
        azi=None,
        polar_reference=None,
        energy=None,
        time=None,
    ):
        label = "cell_tally"
        type_ = TALLY_CELL
        super().__init__(
            label, type_, name, scores, mu, azi, polar_reference, energy, time
        )

        # Attach cell and attach tally to the cell
        self.cell = cell
        cell.N_tally += 1
        cell.tally_IDs.append(self.ID)
        cell.tallies.append(self)

        # Allocate the bins
        N_mu = len(self.mu) - 1
        N_azi = len(self.azi) - 1
        N_energy = len(self.energy) - 1
        N_time = len(self.time) - 1
        N_score = len(self.scores)
        #
        self.bin = np.zeros((N_mu, N_azi, N_energy, N_time, N_score))
        self.bin_sum = np.zeros_like(self.bin)
        self.bin_sum_square = np.zeros_like(self.bin)

        # Set the strides
        self.stride_time = N_score
        self.stride_energy = N_score * N_time
        self.stride_azi = N_score * N_time * N_energy
        self.stride_mu = N_score * N_time * N_energy * N_azi

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Cell: {self.cell.name}\n"
        text += super()._phasespace_filter_text()
        text += f"  - Bin shape (mu, azi, energy, time, score): {self.bin.shape} \n"
        return text


# ======================================================================================
# Surface tally
# ======================================================================================


class TallySurface(TallyBase):
    def __init__(
        self,
        name=None,
        surface=None,
        scores=None,
        mu=None,
        azi=None,
        polar_reference=None,
        energy=None,
        time=None,
    ):
        label = "surface_tally"
        type_ = TALLY_SURFACE
        super().__init__(
            label, type_, name, scores, mu, azi, polar_reference, energy, time
        )

        # Set surface and attach tally to the surface
        self.surface = surface
        surface.N_tally += 1
        surface.tally_IDs.append(self.ID)
        surface.tallies.append(self)

        # Allocate the bins
        N_mu = len(self.mu) - 1
        N_azi = len(self.azi) - 1
        N_energy = len(self.energy) - 1
        N_time = len(self.time) - 1
        N_score = len(self.scores)
        #
        self.bin = np.zeros((N_mu, N_azi, N_energy, N_time, N_score))
        self.bin_sum = np.zeros_like(self.bin)
        self.bin_sum_square = np.zeros_like(self.bin)

        # Set the strides
        self.stride_time = N_score
        self.stride_energy = N_score * N_time
        self.stride_azi = N_score * N_time * N_energy
        self.stride_mu = N_score * N_time * N_energy * N_azi

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Surface: {self.surface.name}\n"
        text += super()._phasespace_filter_text()
        text += f"  - Bin shape (mu, azi, energy, time, score): {self.bin.shape} \n"
        return text


# ======================================================================================
# Mesh tally
# ======================================================================================


class TallyMesh(TallyBase):
    def __init__(
        self,
        name=None,
        mesh=None,
        scores=None,
        mu=None,
        azi=None,
        polar_reference=None,
        energy=None,
    ):
        label = "mesh_tally"
        type_ = TALLY_MESH

        self.mesh = mesh
        if mesh.type == MESH_UNIFORM:
            self.time = np.linspace(mesh.t0, mesh.t0 + mesh.Nt * mesh.dt, mesh.Nt + 1)
        elif mesh.type == MESH_STRUCTURED:
            self.time = mesh.t

        super().__init__(
            label, type_, name, scores, mu, azi, polar_reference, energy, self.time
        )

        # Allocate the bins
        N_mu = len(self.mu) - 1
        N_azi = len(self.azi) - 1
        N_energy = len(self.energy) - 1
        N_score = len(self.scores)
        #
        self.bin = np.zeros((N_mu, N_azi, N_energy, mesh.Nt, mesh.Nx, mesh.Ny, mesh.Nz, N_score))
        self.bin_sum = np.zeros_like(self.bin)
        self.bin_sum_square = np.zeros_like(self.bin)

        # Set the strides
        self.stride_z = N_score
        self.stride_y = N_score * mesh.Nz
        self.stride_x = N_score * mesh.Nz * mesh.Ny
        self.stride_time = N_score * mesh.Nz * mesh.Ny * mesh.Nx
        self.stride_energy = N_score * mesh.Nz * mesh.Ny * mesh.Nx * mesh.Nt
        self.stride_azi = N_score * mesh.Nz * mesh.Ny * mesh.Nx * mesh.Nt * N_energy
        self.stride_mu = N_score * mesh.Nz * mesh.Ny * mesh.Nx * mesh.Nt * N_energy * N_azi

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Mesh: {mesh_module.decode_type(self.mesh.type)} (ID {self.mesh.ID})\n"
        text += super()._phasespace_filter_text()
        text += f"  - Bin shape (mu, azi, energy, time, x, y, z, score): {self.bin.shape} \n"
        return text
