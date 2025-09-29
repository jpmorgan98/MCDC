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

from typing import get_origin, get_args, Union, Annotated

def check_type(value, hint) -> bool:
    """
    Recursively validate that a value matches a given type hint.

    Parameters
    ----------
    value : Any
        The runtime value to check.
    hint : Any
        The type annotation (from typing) to validate against.
        Examples: int, list[str], dict[str, int], tuple[int, ...], Optional[str].

    Returns
    -------
    bool
        True if the value conforms to the type hint, False otherwise.

    Supported type hints
    --------------------
    - Builtins: int, str, float, etc.
    - Generic containers: list[T], set[T], dict[K, V], tuple[T1, T2], tuple[T, ...]
    - Unions (including Optional[T])
    - Annotated[T, ...] (base type T is checked)
    - Fallback: if type cannot be checked, returns True (best-effort).
    """
    origin = get_origin(hint)

    # Case 1: Simple builtin types (int, str, custom classes, etc.)
    if origin is None:
        try:
            return isinstance(value, hint)
        except TypeError:
            # For special forms like typing.Any
            return True

    # Case 2: list[T]
    if origin is list:
        (t,) = get_args(hint)
        return isinstance(value, list) and all(check_type(x, t) for x in value)

    # Case 3: set[T]
    if origin is set:
        (t,) = get_args(hint)
        return isinstance(value, set) and all(check_type(x, t) for x in value)

    # Case 4: dict[K, V]
    if origin is dict:
        kt, vt = get_args(hint)
        return (isinstance(value, dict)
                and all(check_type(k, kt) and check_type(v, vt) for k, v in value.items()))

    # Case 5: tuple[T1, T2] or tuple[T, ...]
    if origin is tuple:
        args = get_args(hint)
        if len(args) == 2 and args[1] is Ellipsis:
            # tuple[T, ...]
            return isinstance(value, tuple) and all(check_type(x, args[0]) for x in value)
        # tuple[T1, T2, ...]
        return (isinstance(value, tuple)
                and len(value) == len(args)
                and all(check_type(x, t) for x, t in zip(value, args)))

    # Case 6: Union[...] (includes Optional[T])
    if origin is Union:
        return any(check_type(value, t) for t in get_args(hint))

    # Case 7: Annotated[T, ...] → check the base type T
    if origin is Annotated:
        return check_type(value, get_args(hint)[0])

    # Fallback: try to check against the origin (e.g., typing.IO, typing.Iterable)
    try:
        return isinstance(value, origin)
    except TypeError:
        return True
