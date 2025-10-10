from __future__ import annotations
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from mcdc.object_.cell import Cell
    from mcdc.object_.surface import Surface

####

import numpy as np

from numpy import float64
from numpy.typing import NDArray
from typing import Annotated
from types import NoneType

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
from mcdc.object_.mesh import MeshBase
from mcdc.object_.base import ObjectPolymorphic
from mcdc.object_.simulation import simulation
from mcdc.print_ import print_1d_array


# ======================================================================================
# Tally base class
# ======================================================================================


class TallyBase(ObjectPolymorphic):
    name: str
    scores: list[int]
    mu: NDArray[float64]
    azi: NDArray[float64]
    polar_reference: Annotated[NDArray[float64], (3,)]
    energy: NDArray[float64]
    time: NDArray[float64]
    filter_direction: bool
    filter_energy: bool
    filter_time: bool
    bin: NDArray[float64]
    bin_sum: NDArray[float64]
    bin_sum_square: NDArray[float64]
    stride_mu: int
    stride_azi: int
    stride_energy: int
    stride_time: int

    def __init__(
        self, type_, name, scores, mu, azi, polar_reference, energy, time
    ):
        super().__init__(type_)

        # Set name
        if name != "":
            self.name = name
        else:
            self.name = f"{self.label}_{self.numba_ID}"

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

    def _use_census_based_tally(self, frequency):
        pass # Implemented in subclass

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
    # Annotations for Numba mode
    label: str = 'cell_tally'
    #
    cell: Cell

    def __init__(
        self,
        name: str = "",
        cell: Cell = None,
        scores: list[str] = ['flux'],
        mu: NDArray[float64] | NoneType = None,
        azi: NDArray[float64] | NoneType = None,
        polar_reference: NDArray[float64] | NoneType = None,
        energy: NDArray[float64] | str | NoneType = None,
        time: NDArray[float64] | NoneType = None,
    ):
        type_ = TALLY_CELL
        super().__init__(
            type_, name, scores, mu, azi, polar_reference, energy, time
        )

        # Attach cell and attach tally to the cell
        self.cell = cell
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

    def _use_census_based_tally(self, frequency):
        self.time = np.zeros(frequency + 1)

        N_mu = len(self.mu) - 1
        N_azi = len(self.azi) - 1
        N_energy = len(self.energy) - 1
        N_score = len(self.scores)

        self.bin = np.zeros((N_mu, N_azi, N_energy, frequency, N_score))
        self.bin_sum = np.zeros_like(self.bin)
        self.bin_sum_square = np.zeros_like(self.bin)

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
    # Annotations for Numba mode
    label: str = 'surface_tally'
    #
    surface: Surface

    def __init__(
        self,
        name: str = "",
        surface: Surface = None,
        scores: list[str] = ['flux'],
        mu: NDArray[float64] | NoneType = None,
        azi: NDArray[float64] | NoneType = None,
        polar_reference: NDArray[float64] | NoneType = None,
        energy: NDArray[float64] | str | NoneType = None,
        time: NDArray[float64] | NoneType = None,
    ):
        type_ = TALLY_SURFACE
        super().__init__(
            type_, name, scores, mu, azi, polar_reference, energy, time
        )

        # Set surface and attach tally to the surface
        self.surface = surface
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

    def _use_census_based_tally(self, frequency):
        self.time = np.zeros(frequency + 1)

        N_mu = len(self.mu) - 1
        N_azi = len(self.azi) - 1
        N_energy = len(self.energy) - 1
        N_score = len(self.scores)

        self.bin = np.zeros((N_mu, N_azi, N_energy, frequency, N_score))
        self.bin_sum = np.zeros_like(self.bin)
        self.bin_sum_square = np.zeros_like(self.bin)

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
    # Annotations for Numba mode
    label: str = 'mesh_tally'
    #
    mesh: MeshBase
    stride_z: int
    stride_y: int
    stride_x: int

    def __init__(
        self,
        name: str = "",
        mesh: MeshBase = None,
        scores: list[str] = ['flux'],
        mu: NDArray[float64] | NoneType = None,
        azi: NDArray[float64] | NoneType = None,
        polar_reference: NDArray[float64] | NoneType = None,
        energy: NDArray[float64] | str | NoneType = None,
    ):
        type_ = TALLY_MESH

        self.mesh = mesh
        if mesh.type == MESH_UNIFORM:
            self.time = np.linspace(mesh.t0, mesh.t0 + mesh.Nt * mesh.dt, mesh.Nt + 1)
        elif mesh.type == MESH_STRUCTURED:
            self.time = mesh.t

        super().__init__(
            type_, name, scores, mu, azi, polar_reference, energy, self.time
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

    def _use_census_based_tally(self, frequency):
        mesh = self.mesh
        self.time = np.zeros(frequency + 1)
        mesh.t = np.zeros(frequency + 1)

        N_mu = len(self.mu) - 1
        N_azi = len(self.azi) - 1
        N_energy = len(self.energy) - 1
        N_score = len(self.scores)

        self.bin = np.zeros((N_mu, N_azi, N_energy, frequency, mesh.Nx, mesh.Ny, mesh.Nz, N_score))
        self.bin_sum = np.zeros_like(self.bin)
        self.bin_sum_square = np.zeros_like(self.bin)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - Mesh: {mesh_module.decode_type(self.mesh.type)} (ID {self.mesh.ID})\n"
        text += super()._phasespace_filter_text()
        text += f"  - Bin shape (mu, azi, energy, time, x, y, z, score): {self.bin.shape} \n"
        return text
