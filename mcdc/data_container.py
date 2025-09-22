from mcdc.constant import DATA_TABLE, DATA_POLYNOMIAL, DATA_MULTIPDF, DATA_MAXWELLIAN
from mcdc.objects import ObjectPolymorphic
from mcdc.prints import print_1d_array
from mcdc.util import cdf_from_pdf


class DataContainer(ObjectPolymorphic):
    """
    Base class for tabulated/parametric data used by reactions.

    Parameters
    ----------
    label : str
        Descriptive label for the data container.
    type_ : int
        Data type code (e.g., `DATA_TABLE`, `DATA_MULTIPDF`).

    Attributes
    ----------
    ID : int
        Internal identifier from :class:`ObjectPolymorphic`.
    type : int
        Data type code.

    See Also
    --------
    Nuclide
        Aggregates reactions and their data.
    ReactionBase
        Abstract base class for reactions.
    ReactionNeutronCapture
        Uses :class:`DataTable` for XS, etc.
    ReactionNeutronElasticScattering
        Uses :class:`DataMultiPDF` for scattering cosine.
    ReactionNeutronFission
        Uses :class:`DataTable`, :class:`DataPolynomial`,
        :class:`DataMultiPDF`, and :class:`DataMaxwellian`.
    """

    def __init__(self, label, type_):
        super().__init__(label, type_)

    def __repr__(self):
        text = "\n"
        text += f"{decode_type(self.type)}\n"
        text += f"  - ID: {self.ID}\n"
        return text


class DataTable(DataContainer):
    """
    1D tabulated function y(x).

    Parameters
    ----------
    x : numpy.ndarray
        Monotonic grid (e.g., energy) for the independent variable.
    y : numpy.ndarray
        Values at each `x`.

    Attributes
    ----------
    x : numpy.ndarray
        Independent variable grid.
    y : numpy.ndarray
        Function values.

    See Also
    --------
    DataPolynomial
        Compact polynomial form for y(x).
    DataMultiPDF
        Piecewise PDF/CDF representation.
    DataMaxwellian
        Maxwellian spectrum parameterization.
    ReactionNeutronFission
        Uses :class:`DataTable` (e.g., yields).
    ReactionNeutronCapture, ReactionNeutronElasticScattering
        May pair with tabulated XS.
    """

    def __init__(self, x, y):
        label = "data_table"
        type_ = DATA_TABLE
        super().__init__(label, type_)

        self.x = x
        self.y = y

    def __repr__(self):
        text = super().__repr__()
        text += f"  - x {print_1d_array(self.x)}\n"
        text += f"  - y {print_1d_array(self.y)}\n"
        return text


class DataPolynomial(DataContainer):
    """
    Polynomial function y(x) = sum_i coeffs[i] * x**i.

    Parameters
    ----------
    coeffs : numpy.ndarray
        Polynomial coefficients ordered by increasing power of x.

    Attributes
    ----------
    coefficients : numpy.ndarray
        Stored polynomial coefficients.

    See Also
    --------
    DataTable
        Alternative tabulated representation.
    ReactionNeutronFission
        May use polynomial yields.
    """

    def __init__(self, coeffs):
        label = "data_polynomial"
        type_ = DATA_POLYNOMIAL
        super().__init__(label, type_)

        self.coefficients = coeffs

    def __repr__(self):
        text = super().__repr__()
        text += f"  - coefficients {print_1d_array(self.coefficients)}\n"
        return text


class DataMultiPDF(DataContainer):
    """
    Piecewise (multi-PDF) representation with per-segment PDFs and derived CDFs.

    Parameters
    ----------
    grid : numpy.ndarray
        Segment boundary grid (e.g., incident energy).
    offset : numpy.ndarray
        Starting index (or pointer) for each segment in `value`/`pdf`.
    value : numpy.ndarray
        Abscissae within segments (e.g., μ or outgoing energy).
    pdf : numpy.ndarray
        Probability density values aligned with `value`.

    Notes
    -----
    - On construction, a CDF is computed segment-wise via :func:`cdf_from_pdf`
      and stored alongside the (possibly normalized) PDF.

    Attributes
    ----------
    grid : numpy.ndarray
        Segment grid.
    offset : numpy.ndarray
        Segment offsets into `value`/`pdf`.
    value : numpy.ndarray
        Local abscissae.
    pdf : numpy.ndarray
        (Re)normalized PDF.
    cdf : numpy.ndarray
        Derived cumulative distribution.

    See Also
    --------
    cdf_from_pdf
        Utility used to compute (pdf, cdf) per segment.
    ReactionNeutronElasticScattering
        Uses :class:`DataMultiPDF` for scattering cosine distributions.
    ReactionNeutronFission
        May use :class:`DataMultiPDF` for prompt/delayed spectra.
    DataTable, DataPolynomial, DataMaxwellian
        Related data parameterizations.
    """

    def __init__(self, grid, offset, value, pdf):
        label = "data_multipdf"
        type_ = DATA_MULTIPDF
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


class DataMaxwellian(DataContainer):
    """
    Maxwellian neutron energy spectrum with optional restriction energy.

    Parameters
    ----------
    restriction_energy : numpy.ndarray or float
        Upper restriction energy ``U`` for the spectrum.
    nuclear_temperature_energy_grid : numpy.ndarray
        Grid for nuclear temperature parameterization.
    nuclear_temperature_value : numpy.ndarray
        Nuclear temperature values corresponding to the grid.

    Attributes
    ----------
    U : numpy.ndarray or float
        Restriction energy.
    T : DataTable
        Temperature parameter as a table over energy.

    See Also
    --------
    DataTable
        Used internally for the temperature curve.
    ReactionNeutronFission
        May use :class:`DataMaxwellian` for prompt/delayed spectra.
    ReactionNeutronElasticScattering
        Alternative spectral forms appear here (contrast with multi-PDF).
    """

    def __init__(
        self,
        restriction_energy,
        nuclear_temperature_energy_grid,
        nuclear_temperature_value,
    ):
        label = "data_maxwellian"
        type_ = DATA_MAXWELLIAN
        super().__init__(label, type_)

        self.U = restriction_energy
        self.T = DataTable(nuclear_temperature_energy_grid, nuclear_temperature_value)

    def __repr__(self):
        text = super().__repr__()
        text += f"  - U {print_1d_array(self.U)}\n"
        text += f"  - T energy_grid {print_1d_array(self.T.x)}\n"
        text += f"  - T {print_1d_array(self.T.y)}\n"
        return text


def decode_type(type_):
    if type_ == DATA_TABLE:
        return "Data (Table)"
    elif type_ == DATA_POLYNOMIAL:
        return "Data (Polynomial function)"
    elif type_ == DATA_MULTIPDF:
        return "Data (Multi PDF)"
    elif type_ == DATA_MAXWELLIAN:
        return "Data (Maxwellian spectrum)"
