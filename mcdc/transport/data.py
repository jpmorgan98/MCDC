from numba import njit

####

import mcdc.mcdc_get as mcdc_get

from mcdc.constant import DATA_POLYNOMIAL, DATA_TABLE
from mcdc.transport.util import find_bin, linear_interpolation


@njit
def evaluate_data(x, data_base, mcdc, data):
    data_type = data_base["child_type"]
    ID = data_base["child_ID"]
    if data_type == DATA_TABLE:
        table = mcdc["table_data"][ID]
        return evaluate_table(x, table, data)
    elif data_type == DATA_POLYNOMIAL:
        polynomial = mcdc["polynomial_data"][ID]
        return evaluate_polynomial(x, polynomial, data)
    else:
        return 0.0


@njit
def evaluate_table(x, table, data):
    grid = mcdc_get.table_data.x_all(table, data)
    idx = find_bin(x, grid)
    x1 = grid[idx]
    x2 = grid[idx + 1]
    y1 = mcdc_get.table_data.y(idx, table, data)
    y2 = mcdc_get.table_data.y(idx + 1, table, data)
    return linear_interpolation(x, x1, x2, y1, y2)


@njit
def evaluate_polynomial(x, polynomial, data):
    coeffs = mcdc_get.polynomial_data.coefficients_all(polynomial, data)
    total = 0.0
    for i in range(len(coeffs)):
        total += coeffs[i] * x**i
    return total
