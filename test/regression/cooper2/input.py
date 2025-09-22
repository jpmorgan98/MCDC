import numpy as np
import mcdc


# =============================================================================
# Set model
# =============================================================================
# A shielding problem based on Problem 2 of [Coper NSE 2001]
# https://ans.tandfonline.com/action/showCitFormats?doi=10.13182/NSE00-34

# Set materials
SigmaT = 5.0
c = 0.8
m_barrier = mcdc.MaterialMG(capture=np.array([SigmaT]), scatter=np.array([[SigmaT * c]]))
SigmaT = 1.0
m_room = mcdc.MaterialMG(capture=np.array([SigmaT]), scatter=np.array([[SigmaT * c]]))

# Set surfaces
sx1 = mcdc.Surface.PlaneX(x=0.0, boundary_condition="reflective")
sx2 = mcdc.Surface.PlaneX(x=2.0)
sx3 = mcdc.Surface.PlaneX(x=2.4)
sx4 = mcdc.Surface.PlaneX(x=4.0, boundary_condition="vacuum")
sy1 = mcdc.Surface.PlaneY(y=0.0, boundary_condition="reflective")
sy2 = mcdc.Surface.PlaneY(y=2.0)
sy3 = mcdc.Surface.PlaneY(y=4.0, boundary_condition="vacuum")

# Set cells
mcdc.Cell(region=+sx1 & -sx2 & +sy1 & -sy2, fill=m_room)
mcdc.Cell(region=+sx1 & -sx4 & +sy2 & -sy3, fill=m_room)
mcdc.Cell(region=+sx3 & -sx4 & +sy1 & -sy2, fill=m_room)
mcdc.Cell(region=+sx2 & -sx3 & +sy1 & -sy2, fill=m_barrier)

# =============================================================================
# Set source
# =============================================================================
# Uniform isotropic source throughout the domain

mcdc.source(x=[0.0, 1.0], y=[0.0, 1.0], isotropic=True)

# =============================================================================
# Set tally, setting, and run mcdc
# =============================================================================

mcdc.Settings(N_particle=50, N_batch=2)
mcdc.implicit_capture()

mcdc.tally.mesh_tally(
    scores=["flux"],
    x=np.linspace(0.0, 4.0, 40),
    y=np.linspace(0.0, 4.0, 40),
)

mcdc.run()
