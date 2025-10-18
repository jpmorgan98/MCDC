# ======================================================================================
# Simulation building blocks
# ======================================================================================

# The simulation
from mcdc.object_.simulation import simulation

settings = simulation.settings

# The objects
from mcdc.object_.cell import Cell, Universe, Lattice
from mcdc.object_.material import Material, MaterialMG
from mcdc.object_.mesh import MeshUniform, MeshStructured
from mcdc.object_.source import Source
from mcdc.object_.surface import Surface
from mcdc.object_.tally import TallyGlobal, TallyCell, TallySurface, TallyMesh

# ======================================================================================
# Runners
# ======================================================================================

from mcdc.main import run

# ======================================================================================
# Will be replaced
# ======================================================================================

"""
from mcdc.input_ import (
    source,
    implicit_capture,
    weighted_emission,
    population_control,
    time_census,
    weight_window,
    iQMC,
    weight_roulette,
    uq,
    reset,
    domain_decomposition,
    make_particle_bank,
    save_particle_bank,
)
from mcdc.main import (
    prepare,
    run,
    visualize,
    recombine_tallies,
)
"""
