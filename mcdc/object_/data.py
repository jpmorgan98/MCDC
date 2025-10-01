from typing import Iterable

####

from mcdc.constant import DATA_TABLE, DATA_POLYNOMIAL
from mcdc.object_.base import ObjectPolymorphic
from mcdc.print_ import print_1d_array


# ======================================================================================
# Data base class
# ======================================================================================


class DataBase(ObjectPolymorphic):
    label: str = 'data'

    def __init__(self, type_):
        super().__init__(type_)

    def __repr__(self):
        text = "\n"
        text += f"{decode_type(self.type)}\n"
        text += f"  - ID: {self.ID}\n"
        return text


def decode_type(type_):
    if type_ == DATA_TABLE:
        return "Data (Table)"
    elif type_ == DATA_POLYNOMIAL:
        return "Data (Polynomial function)"


# ======================================================================================
# Table data
# ======================================================================================


class DataTable(DataBase):
    label: str = 'table_data'

    # Annotations for Numba mode
    x: Iterable[float]
    y: Iterable[float]

    def __init__(self, x, y):
        type_ = DATA_TABLE
        super().__init__(type_)

        self.x = x
        self.y = y

    def __repr__(self):
        text = super().__repr__()
        text += f"  - x {print_1d_array(self.x)}\n"
        text += f"  - y {print_1d_array(self.y)}\n"
        return text


# ======================================================================================
# Polynomial data
# ======================================================================================


class DataPolynomial(DataBase):
    label: str = 'polynomial_data'

    # Annotations for Numba mode
    coefficients: Iterable[float]

    def __init__(self, coeffs):
        type_ = DATA_POLYNOMIAL
        super().__init__(type_)

        self.coefficients = coeffs

    def __repr__(self):
        text = super().__repr__()
        text += f"  - coefficients {print_1d_array(self.coefficients)}\n"
        return text
