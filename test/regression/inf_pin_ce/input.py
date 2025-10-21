import numpy as np
import os
import mcdc


# Set the XS library directory
os.environ["MCDC_XSLIB"] = os.path.dirname(os.getcwd()) + "/dummy_data"

# =============================================================================
# Set model
# =============================================================================

# Set materials

fuel = mcdc.Material(
    nuclide_composition={
        "dummy_nuclide_4": 0.0005581658948833916 * 7e-2,
        "dummy_nuclide_5": 0.022404594715383263 * 7e-2,
        "dummy_nuclide_3": 0.045831301393656466,
    }
)

water = mcdc.Material(
    nuclide_composition={
        "dummy_nuclide_2": 0.0001357003217727274 * 100.0,
        "dummy_nuclide_1": 0.0684556951587359,
        "dummy_nuclide_3": 0.032785655643293984,
    }
)

# Set surfaces
cy = mcdc.Surface.CylinderZ(center=[0.0, 0.0], radius=0.45720)
pitch = 1.25984
x1 = mcdc.Surface.PlaneX(x=-pitch / 2, boundary_condition="reflective")
x2 = mcdc.Surface.PlaneX(x=pitch / 2, boundary_condition="reflective")
y1 = mcdc.Surface.PlaneY(y=-pitch / 2, boundary_condition="reflective")
y2 = mcdc.Surface.PlaneY(y=pitch / 2, boundary_condition="reflective")

# Set cells
mcdc.Cell(region=-cy & +x1 & -x2 & +y1 & -y2, fill=fuel)
mcdc.Cell(region=+cy & +x1 & -x2 & +y1 & -y2, fill=water)

# =============================================================================
# Set source
# =============================================================================

mcdc.Source(
    x=[-pitch / 2, pitch / 2],
    y=[-pitch / 2, pitch / 2],
    isotropic=True,
    energy=1E6,
)

# =============================================================================
# Set tallies, settings, and run MC/DC
# =============================================================================

# Tallies
mcdc.TallyGlobal(
    scores=["flux", "density"],
    energy=np.loadtxt("energy_grid.txt"),
    time=np.insert(np.logspace(-8, 2, 50), 0, 0.0),
)

# Settings
mcdc.settings.N_particle = 100
mcdc.settings.N_batch = 2
mcdc.settings.active_bank_buffer = 1000

# Run
mcdc.run()
