import numpy as np


def cdf_from_pdf(offset, value, pdf):
    """
    Construct cumulative distribution function (CDF) from a piecewise-linear
    probability density function (PDF), handling multiple disjoint segments.

    Parameters
    ----------
    offset : array_like of int
        Segment boundaries in `pdf` and `value`.
        For each i, the segment is pdf[offset[i]:offset[i+1]].
        The last entry may point to len(pdf).
    value : array_like of float
        Grid of x-values corresponding to pdf.
        Must have same length as pdf.
    pdf : array_like of float
        Probability density values at grid points.
        Will be modified in-place (normalized by segment).

    Returns
    -------
    pdf : ndarray
        Normalized PDF (each segment integrates to 1).
    cdf : ndarray
        CDF values at the same grid points, normalized per segment.

    Notes
    -----
    - Integration uses the trapezoidal rule on each segment.
    - Each segment [offset[i], offset[i+1]) is normalized separately.
    - The last segment is defined from offset[-1] to len(pdf).

    Example
    -------
    >>> value = np.linspace(0, 1, 6)
    >>> pdf = np.ones_like(value)     # flat pdf
    >>> offset = [0]
    >>> pdf_norm, cdf = cdf_from_pdf(offset, value, pdf)
    >>> cdf[-1]
    1.0
    """
    cdf = np.zeros_like(pdf)

    for i in range(len(offset)):
        start = offset[i]
        end = offset[i + 1] if i < len(offset) - 1 else len(pdf)

        # Build CDF incrementally with trapezoidal integration
        for idx in range(start, end - 1):
            cdf[idx + 1] = (
                cdf[idx]
                + (pdf[idx] + pdf[idx + 1]) * (value[idx + 1] - value[idx]) * 0.5
            )

        # Normalize this segment so CDF ends at 1
        norm = cdf[end - 1]
        pdf[start:end] /= norm
        cdf[start:end] /= norm

    return pdf, cdf


def is_sorted(a):
    return np.all(a[:-1] <= a[1:])
