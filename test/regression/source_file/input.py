import numpy as np
from mpi4py import MPI

import mcdc

# =============================================================================
# Set source particles and create the source particle file
# (only master, rank 0)
# =============================================================================

if MPI.COMM_WORLD.Get_rank() == 0:
    rng = np.random.default_rng(seed=7)

    N = 500
    bank = mcdc.make_particle_bank(N)

    for i in range(N):
        particle = bank[i]
        particle["x"] = rng.random() * 5.0
        particle["y"] = rng.random()
        particle["z"] = rng.random()
        particle["t"] = rng.random() * 5.0
        particle["ux"] = 1.0  # All going right
        particle["uy"] = 0.0
        particle["uz"] = 0.0
        particle["E"] = 1e6  # Arbitrary
        particle["type"] = 0

    mcdc.save_particle_bank(bank, "source_particles")
MPI.COMM_WORLD.Barrier()


# =============================================================================
# Set model
# =============================================================================

# Set materials
m = mcdc.MaterialMG(capture=np.array([0.1]), scatter=np.array([[0.9]]))

# Set surfaces
s1 = mcdc.Surface.PlaneX(x=0.0, boundary_condition="vacuum")
s2 = mcdc.Surface.PlaneX(x=5.0, boundary_condition="vacuum")

# Set cells
mcdc.Cell(region=+s1 & -s2, fill=m)


# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

settings = mcdc.Settings()
settings.set_source_file("source_particles.h5")

mcdc.tally.mesh_tally(
    scores=["flux"],
    x=np.linspace(0.0, 5.0, 51),
    t=np.linspace(0.0, 5.0, 51),
)

mcdc.run()
