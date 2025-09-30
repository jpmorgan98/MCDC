from typing import Iterable

####

from mcdc.constant import DISTRIBUTION_MULTIPDF, DISTRIBUTION_MAXWELLIAN
from mcdc.object_.base import ObjectPolymorphic
from mcdc.object_.data import DataTable
from mcdc.print_ import print_1d_array
from mcdc.object_.util import cdf_from_pdf


# ======================================================================================
# Distribution base class
# ======================================================================================


class DistributionBase(ObjectPolymorphic):
    def __init__(self, label, type_):
        super().__init__(label, type_)

    def __repr__(self):
        text = "\n"
        text += f"{decode_type(self.type)}\n"
        text += f"  - ID: {self.ID}\n"
        return text


def decode_type(type_):
    if type_ == DISTRIBUTION_MULTIPDF:
        return "Data (Multi PDF)"
    elif type_ == DISTRIBUTION_MAXWELLIAN:
        return "Data (Maxwellian spectrum)"


# ======================================================================================
# MultiPDF distribution
# ======================================================================================


class DistributionMultiPDF(DistributionBase):
    # Annotations for Numba mode
    grid: Iterable[float]
    offset: Iterable[float]
    value: Iterable[float]
    pdf: Iterable[float]
    cdf: Iterable[float]

    def __init__(self, grid, offset, value, pdf):
        label = "data_multipdf"
        type_ = DISTRIBUTION_MULTIPDF
        super().__init__(label, type_)

        self.grid = grid
        self.offset = offset
        self.value = value
        self.pdf = pdf

        self.pdf, self.cdf = cdf_from_pdf(offset, value, pdf)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - grid {print_1d_array(self.grid)}\n"
        text += f"  - offset {print_1d_array(self.offset)}\n"
        text += f"  - value {print_1d_array(self.value)}\n"
        text += f"  - pdf {print_1d_array(self.pdf)}\n"
        return text


# ======================================================================================
# Maxwellian distribution
# ======================================================================================


class DistributionMaxwellian(DistributionBase):
    # Annotations for Numba mode
    U: float
    T: DataTable

    def __init__(
        self,
        restriction_energy,
        nuclear_temperature_energy_grid,
        nuclear_temperature_value,
    ):
        label = "data_maxwellian"
        type_ = DISTRIBUTION_MAXWELLIAN
        super().__init__(label, type_)

        self.U = restriction_energy
        self.T = DataTable(nuclear_temperature_energy_grid, nuclear_temperature_value)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - U {print_1d_array(self.U)}\n"
        text += f"  - T energy_grid {print_1d_array(self.T.x)}\n"
        text += f"  - T {print_1d_array(self.T.y)}\n"
        return text
