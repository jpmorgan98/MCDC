from numpy import float64
from numpy.typing import NDArray

####

from mcdc.constant import DISTRIBUTION_MULTIPDF, DISTRIBUTION_MAXWELLIAN, DISTRIBUTION_PDF, DISTRIBUTION_PMF
from mcdc.object_.base import ObjectPolymorphic
from mcdc.object_.data import DataTable
from mcdc.print_ import print_1d_array
from mcdc.object_.util import cdf_from_pdf, multi_cdf_from_pdf, cmf_from_pmf


# ======================================================================================
# Distribution base class
# ======================================================================================


class DistributionBase(ObjectPolymorphic):
    def __init__(self, type_):
        super().__init__(type_)

    def __repr__(self):
        text = "\n"
        text += f"{decode_type(self.type)}\n"
        text += f"  - ID: {self.ID}\n"
        return text


def decode_type(type_):
    if type_ == DISTRIBUTION_PDF:
        return "Distribution (PDF)"
    elif type_ == DISTRIBUTION_PMF:
        return "Distribution (PMF)"
    elif type_ == DISTRIBUTION_MULTIPDF:
        return "Distribution (Multi PDF)"
    elif type_ == DISTRIBUTION_MAXWELLIAN:
        return "Distribution (Maxwellian spectrum)"


# ======================================================================================
# PDF distribution
# ======================================================================================

class DistributionPDF(DistributionBase):
    # Annotations for Numba mode
    label: str = 'pdf_distribution'
    #
    value: NDArray[float64]
    pdf: NDArray[float64]
    cdf: NDArray[float64]

    def __init__(self, value, pdf):
        type_ = DISTRIBUTION_PDF
        super().__init__(type_)

        self.value = value
        self.pdf = pdf

        self.pdf, self.cdf = cdf_from_pdf(value, pdf)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - value {print_1d_array(self.value)}\n"
        text += f"  - pdf {print_1d_array(self.pdf)}\n"
        return text


# ======================================================================================
# PMF distribution
# ======================================================================================

class DistributionPMF(DistributionBase):
    # Annotations for Numba mode
    label: str = 'pmf_distribution'
    #
    value: NDArray[float64]
    pmf: NDArray[float64]
    cmf: NDArray[float64]

    def __init__(self, value, pmf):
        type_ = DISTRIBUTION_PMF
        super().__init__(type_)

        self.value = value
        self.pmf = pmf

        self.pmf, self.cmf = cmf_from_pmf(value, pmf)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - value {print_1d_array(self.value)}\n"
        text += f"  - pmf {print_1d_array(self.pmf)}\n"
        return text



# ======================================================================================
# MultiPDF distribution
# ======================================================================================


class DistributionMultiPDF(DistributionBase):
    # Annotations for Numba mode
    label: str = 'multipdf_distribution'
    #
    grid: NDArray[float64]
    offset: NDArray[float64]
    value: NDArray[float64]
    pdf: NDArray[float64]
    cdf: NDArray[float64]

    def __init__(self, grid, offset, value, pdf):
        type_ = DISTRIBUTION_MULTIPDF
        super().__init__(type_)

        self.grid = grid
        self.offset = offset
        self.value = value
        self.pdf = pdf

        self.pdf, self.cdf = multi_cdf_from_pdf(offset, value, pdf)

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
    label: str = 'maxwellian_distribution'
    #
    U: float
    T: DataTable

    def __init__(
        self,
        restriction_energy,
        nuclear_temperature_energy_grid,
        nuclear_temperature_value,
    ):
        type_ = DISTRIBUTION_MAXWELLIAN
        super().__init__(type_)

        self.U = restriction_energy
        self.T = DataTable(nuclear_temperature_energy_grid, nuclear_temperature_value)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - U {print_1d_array(self.U)}\n"
        text += f"  - T energy_grid {print_1d_array(self.T.x)}\n"
        text += f"  - T {print_1d_array(self.T.y)}\n"
        return text
