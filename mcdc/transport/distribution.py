import math

from numba import njit

####

import mcdc.mcdc_get as mcdc_get
import mcdc.transport.kernel as kernel

from mcdc.constant import DISTRIBUTION_MAXWELLIAN, DISTRIBUTION_MULTIPDF, PI
from mcdc.transport.data import evaluate_table
from mcdc.transport.util import find_bin


@njit
def sample_distribution(x, distribution_type, index, rng_state, mcdc, data, scale=False):
    if distribution_type == DISTRIBUTION_MULTIPDF:
        multipdf = mcdc["distribution_multipdfs"][index]
        return sample_multipdf(x, rng_state, multipdf, data, scale)
    elif distribution_type == DISTRIBUTION_MAXWELLIAN:
        maxwellian = mcdc["distribution_maxwellians"][index]
        return sample_maxwellian(x, rng_state, maxwellian, mcdc, data)
    else:
        return 0.0


@njit
def sample_multipdf(x, rng_state, multipdf, data, scale=False):
    grid = mcdc_get.multipdf.grid_all(multipdf, data)

    # Edge cases
    if x < grid[0]:
        idx = 0
        scale = False
    elif x > grid[-1]:
        idx = len(grid) - 1
        scale = False
    else:
        # Interpolation factor
        idx = binary_search(x, grid)
        x0 = grid[idx]
        x1 = grid[idx + 1]
        f = (x - x0) / (x1 - x0)

        # Min and max values for scaling
        val_min = 0.0
        val_max = 1.0
        if scale:
            # First table
            start = int(mcdc_get.multipdf.offset(idx, multipdf, data))
            end = int(mcdc_get.multipdf.offset(idx + 1, multipdf, data))
            val0_min = mcdc_get.multipdf.value(start, multipdf, data)
            val0_max = mcdc_get.multipdf.value(end - 1, multipdf, data)

            # Second table
            start = end
            if idx + 2 == len(grid):
                end = multipdf["N_value"]
            else:
                end = int(mcdc_get.multipdf.offset(idx + 2, multipdf, data))
            val1_min = mcdc_get.multipdf.value(start, multipdf, data)
            val1_max = mcdc_get.multipdf.value(end - 1, multipdf, data)

            # Both
            val_min = val0_min + f * (val1_min - val0_min)
            val_max = val0_max + f * (val1_max - val0_max)

        # Sample which table to choose
        if kernel.rng(rng_state) > f:
            idx += 1

    # Get the table range
    start = int(mcdc_get.multipdf.offset(idx, multipdf, data))
    if idx + 1 == len(grid):
        end = multipdf["N_value"]
    else:
        end = int(mcdc_get.multipdf.offset(idx + 1, multipdf, data))
    size = end - start

    # The CDF
    cdf = mcdc_get.multipdf.cdf_chunk(start, size, multipdf, data)

    # Generate random numbers
    xi = kernel.rng(rng_state)

    # Sample bin index
    idx = binary_search(xi, cdf)
    c = cdf[idx]

    # Get the other values
    idx += start  # Apply the offset as these are not chunk-extracted like the cdf
    p0 = mcdc_get.multipdf.pdf(idx, multipdf, data)
    p1 = mcdc_get.multipdf.pdf(idx + 1, multipdf, data)
    val0 = mcdc_get.multipdf.value(idx, multipdf, data)
    val1 = mcdc_get.multipdf.value(idx + 1, multipdf, data)

    m = (p1 - p0) / (val1 - val0)
    if m == 0.0:
        sample = val0 + (xi - c) / p0
    else:
        sample = val0 + 1.0 / m * (math.sqrt(p0**2 + 2 * m * (xi - c)) - p0)

    if not scale:
        return sample

    # Scale against the bounds
    val_low = mcdc_get.multipdf.value(start, multipdf, data)
    val_high = mcdc_get.multipdf.value(end - 1, multipdf, data)
    return val_min + (sample - val_low) / (val_high - val_low) * (val_max - val_min)


@njit
def sample_maxwellian(x, rng_state, maxwellian, mcdc, data):
    # Get nuclear temperature
    table = mcdc["data_tables"][maxwellian["T_index"]]
    T = evaluate_table(x, table, data)
    U = maxwellian["U"]

    # Rejection sampling
    while True:
        xi1 = kernel.rng(rng_state)
        xi2 = kernel.rng(rng_state)
        xi3 = kernel.rng(rng_state)
        cos = math.cos(0.5 * PI * xi3)
        cos_square = cos * cos
        sample = -T * (math.log(xi1) + math.log(xi2) * cos_square)

        # Accept sample
        if sample >= 0.0 and sample <= x - U:
            break

    return sample
