import math

from numba import njit

####

import mcdc.mcdc_get as mcdc_get
import mcdc.transport.rng as rng

from mcdc.constant import DISTRIBUTION_MAXWELLIAN, DISTRIBUTION_MULTITABLE, PI
from mcdc.transport.data import evaluate_table
from mcdc.transport.util import find_bin, linear_interpolation


@njit
def sample_uniform(low, high, rng_state):
    return low + rng.lcg(rng_state) * (high - low)


@njit
def sample_isotropic_direction(rng_state):
    # Sample polar cosine and azimuthal angle uniformly
    mu = 2.0 * rng.lcg(rng_state) - 1.0
    azi = 2.0 * PI * rng.lcg(rng_state)

    # Convert to Cartesian coordinates
    c = (1.0 - mu**2) ** 0.5
    y = math.cos(azi) * c
    z = math.sin(azi) * c
    x = mu
    return x, y, z


@njit
def sample_distribution(x, distribution, rng_state, mcdc, data, scale=False):
    distribution_type = distribution["child_type"]
    ID = distribution["child_ID"]
    if distribution_type == DISTRIBUTION_MULTITABLE:
        multi_table = mcdc["multi_table_distributions"][ID]
        return sample_multi_table(x, rng_state, multi_table, data, scale)
    elif distribution_type == DISTRIBUTION_MAXWELLIAN:
        maxwellian = mcdc["maxwellian_distributions"][ID]
        return sample_maxwellian(x, rng_state, maxwellian, mcdc, data)
    else:
        return 0.0


@njit
def sample_tabulated(table, rng_state, data):
    xi = rng.lcg(rng_state)
    idx = find_bin(xi, mcdc_get.tabulated_distribution.cdf_all(table, data))
    cdf_low = mcdc_get.tabulated_distribution.cdf(idx, table, data)
    cdf_high = mcdc_get.tabulated_distribution.cdf(idx + 1, table, data)
    value_low = mcdc_get.tabulated_distribution.value(idx, table, data)
    value_high = mcdc_get.tabulated_distribution.value(idx + 1, table, data)
    return linear_interpolation(xi, cdf_low, cdf_high, value_low, value_high)


@njit
def sample_pmf(pmf, rng_state, data):
    xi = rng.lcg(rng_state)
    idx = find_bin(xi, mcdc_get.pmf_distribution.cmf_all(pmf, data))
    return mcdc_get.pmf_distribution.value(idx, pmf, data)


@njit
def sample_white_direction(nx, ny, nz, rng_state):
    # Sample polar cosine
    mu = math.sqrt(rng.lcg(rng_state))

    # Sample azimuthal direction
    azi = 2.0 * PI * rng.lcg(rng_state)
    cos_azi = math.cos(azi)
    sin_azi = math.sin(azi)
    Ac = (1.0 - mu**2) ** 0.5

    if nz != 1.0:
        B = (1.0 - nz**2) ** 0.5
        C = Ac / B

        x = nx * mu + (nx * nz * cos_azi - ny * sin_azi) * C
        y = ny * mu + (ny * nz * cos_azi + nx * sin_azi) * C
        z = nz * mu - cos_azi * Ac * B

    # If dir = 0i + 0j + k, interchange z and y in the formula
    else:
        B = (1.0 - ny**2) ** 0.5
        C = Ac / B

        x = nx * mu + (nx * ny * cos_azi - nz * sin_azi) * C
        z = nz * mu + (nz * ny * cos_azi + nx * sin_azi) * C
        y = ny * mu - cos_azi * Ac * B
    return x, y, z


@njit
def sample_multi_table(x, rng_state, multi_table, data, scale=False):
    grid = mcdc_get.multi_table_distribution.grid_all(multi_table, data)

    # Edge cases
    if x < grid[0]:
        idx = 0
        scale = False
    elif x > grid[-1]:
        idx = len(grid) - 1
        scale = False
    else:
        # Interpolation factor
        idx = find_bin(x, grid)
        x0 = grid[idx]
        x1 = grid[idx + 1]
        f = (x - x0) / (x1 - x0)

        # Min and max values for scaling
        val_min = 0.0
        val_max = 1.0
        if scale:
            # First table
            start = int(mcdc_get.multi_table_distribution.offset(idx, multi_table, data))
            end = int(mcdc_get.multi_table_distribution.offset(idx + 1, multi_table, data))
            val0_min = mcdc_get.multi_table_distribution.value(start, multi_table, data)
            val0_max = mcdc_get.multi_table_distribution.value(end - 1, multi_table, data)

            # Second table
            start = end
            if idx + 2 == len(grid):
                end = multi_table["value_length"]
            else:
                end = int(
                    mcdc_get.multi_table_distribution.offset(idx + 2, multi_table, data)
                )
            val1_min = mcdc_get.multi_table_distribution.value(start, multi_table, data)
            val1_max = mcdc_get.multi_table_distribution.value(end - 1, multi_table, data)

            # Both
            val_min = val0_min + f * (val1_min - val0_min)
            val_max = val0_max + f * (val1_max - val0_max)

        # Sample which table to choose
        if rng.lcg(rng_state) > f:
            idx += 1

    # Get the table range
    start = int(mcdc_get.multi_table_distribution.offset(idx, multi_table, data))
    if idx + 1 == len(grid):
        end = multi_table["value_length"]
    else:
        end = int(mcdc_get.multi_table_distribution.offset(idx + 1, multi_table, data))
    size = end - start

    # The CDF
    cdf = mcdc_get.multi_table_distribution.cdf_chunk(start, size, multi_table, data)

    # Generate random numbers
    xi = rng.lcg(rng_state)

    # Sample bin index
    idx = find_bin(xi, cdf)
    c = cdf[idx]

    # Get the other values
    idx += start  # Apply the offset as these are not chunk-extracted like the cdf
    p0 = mcdc_get.multi_table_distribution.pdf(idx, multi_table, data)
    p1 = mcdc_get.multi_table_distribution.pdf(idx + 1, multi_table, data)
    val0 = mcdc_get.multi_table_distribution.value(idx, multi_table, data)
    val1 = mcdc_get.multi_table_distribution.value(idx + 1, multi_table, data)

    m = (p1 - p0) / (val1 - val0)
    if m == 0.0:
        sample = val0 + (xi - c) / p0
    else:
        sample = val0 + 1.0 / m * (math.sqrt(p0**2 + 2 * m * (xi - c)) - p0)

    if not scale:
        return sample

    # Scale against the bounds
    val_low = mcdc_get.multi_table_distribution.value(start, multi_table, data)
    val_high = mcdc_get.multi_table_distribution.value(end - 1, multi_table, data)
    return val_min + (sample - val_low) / (val_high - val_low) * (val_max - val_min)


@njit
def sample_maxwellian(x, rng_state, maxwellian, mcdc, data):
    # Get nuclear temperature
    table = mcdc["table_data"][maxwellian["T_ID"]]
    T = evaluate_table(x, table, data)
    U = maxwellian["U"]

    # Rejection sampling
    while True:
        xi1 = rng.lcg(rng_state)
        xi2 = rng.lcg(rng_state)
        xi3 = rng.lcg(rng_state)
        cos = math.cos(0.5 * PI * xi3)
        cos_square = cos * cos
        sample = -T * (math.log(xi1) + math.log(xi2) * cos_square)

        # Accept sample
        if sample >= 0.0 and sample <= x - U:
            break

    return sample
