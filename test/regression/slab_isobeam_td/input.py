import numpy as np

import mcdc


# =============================================================================
# Set model
# =============================================================================
# Finite homogeneous pure-absorbing slab

# Set materials
m = mcdc.MaterialMG(capture=np.array([1.0]))

# Set surfaces
s1 = mcdc.Surface.PlaneX(x=0.0, boundary_condition="vacuum")
s2 = mcdc.Surface.PlaneX(x=5.0, boundary_condition="vacuum")

# Set cells
mcdc.Cell(region=+s1 & -s2, fill=m)

# =============================================================================
# Set source
# =============================================================================
# Isotropic beam from left-end

mcdc.source(point=[1e-10, 0.0, 0.0], time=[0.0, 5.0], white_direction=[1.0, 0.0, 0.0])

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

settings = mcdc.Settings(N_particle=100, N_batch=2)

mcdc.tally.mesh_tally(
    scores=["flux"],
    x=np.linspace(0.0, 5.0, 51),
    t=np.linspace(0.0, 5.0, 51),
)

mcdc.run()
