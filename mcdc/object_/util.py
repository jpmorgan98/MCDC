import numpy as np

from typing import get_origin, get_args, Union, Annotated


def cmf_from_pmf(value, pmf):
    cmf = np.zeros(len(pmf) + 1)

    # Build CMF incrementally
    total = 0.0
    for idx in range(len(pmf)):
        total += pmf[idx]
        cmf[idx + 1] = total

    # Normalize this segment so CDF ends at 1
    norm = cmf[-1]
    pmf /= norm
    cmf /= norm

    return pmf, cmf

def cdf_from_pdf(value, pdf):
    cdf = np.zeros_like(pdf)

    # Build CDF incrementally with trapezoidal integration
    for idx in range(len(pdf) - 1):
        cdf[idx + 1] = (
            cdf[idx]
            + (pdf[idx] + pdf[idx + 1]) * (value[idx + 1] - value[idx]) * 0.5
        )

    # Normalize this segment so CDF ends at 1
    norm = cdf[-1]
    pdf /= norm
    cdf /= norm

    return pdf, cdf

def multi_cdf_from_pdf(offset, value, pdf):
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


# ======================================================================================
# Typing
# ======================================================================================

import re
import numpy as np
from typing import get_origin, get_args, Union, Annotated

def _name_from_str(s: str) -> str:
    s = _strip_prefixes(s)
    # strip generic args like "NDArray[float64]" → "NDArray"
    s = s.split('[', 1)[0]
    return s.split('.')[-1].strip()

def _mro_name_match(value, want: str) -> bool:
    """Subclass-friendly match without resolving: compare wanted name to any base in MRO."""
    want_name = _name_from_str(want)
    return any(base.__name__ == want_name for base in value.__class__.mro())

# ---------- helpers for STRING annotations ----------
_ANN_RE = re.compile(r'^\s*(?:typing\.)?Annotated\[(.*)\]\s*$')

def _split_top_level(s: str, sep: str = ',', brackets: str = '[]()') -> list[str]:
    out, buf, depth = [], [], 0
    opens = set(brackets[::2]); closes = set(brackets[1::2])
    pairs = dict(zip(brackets[1::2], brackets[::2]))
    for ch in s:
        if ch in opens: depth += 1
        elif ch in closes: depth -= 1
        if ch == sep and depth == 0:
            out.append(''.join(buf).strip()); buf = []
        else:
            buf.append(ch)
    if buf: out.append(''.join(buf).strip())
    return out

def _strip_prefixes(s: str) -> str:
    # normalize common module prefixes used in annotations
    return (s.replace('typing.', '')
             .replace('numpy.typing.', '')
             .replace('numpy.', '')
             .replace('np.', ''))

def _parse_annotated_str(hint_str: str):
    """
    If hint_str is 'Annotated[ ... ]', return (base_str, meta_list) else None.
    meta_list items remain raw strings (no eval).
    """
    m = _ANN_RE.match(_strip_prefixes(hint_str))
    if not m:
        return None
    inner = m.group(1)
    parts = _split_top_level(inner, sep=',')
    if not parts:
        return None
    base = parts[0].strip()
    meta = [p.strip() for p in parts[1:]]
    return base, meta

def _shape_tuple_from_str(s: str):
    """
    Parse '(3,)', '(None, 3)', '(2,3,4)' → tuple[int|None, ...] or None if not a shape.
    """
    s = s.strip()
    if not (s.startswith('(') and s.endswith(')')):
        return None
    body = s[1:-1].strip()
    if not body:
        return ()
    items = _split_top_level(body, sep=',')
    out = []
    for it in items:
        it = it.strip()
        if it == '':
            continue  # allow trailing comma
        if it == 'None':
            out.append(None)
        else:
            try:
                out.append(int(it))
            except ValueError:
                return None
    return tuple(out)

def _is_ndarray_base_str(base_str: str) -> bool:
    base_norm = _strip_prefixes(base_str)
    return base_norm.startswith('NDArray[') or base_norm.startswith('ndarray[')

def _extract_ndarray_dtype_key_from_str(base_str: str) -> str | None:
    base_norm = _strip_prefixes(base_str)
    if '[' not in base_norm or ']' not in base_norm:
        return None
    inside = base_norm[base_norm.find('[')+1: base_norm.rfind(']')].strip()
    return _strip_prefixes(inside)  # e.g. 'float' or 'float64'

def _dtype_matches(arr: np.ndarray, dtype_key: str | None) -> bool:
    if dtype_key is None:
        return True
    key = dtype_key.lower()
    if key == 'float':
        return np.issubdtype(arr.dtype, np.floating)
    if key == 'int':
        return np.issubdtype(arr.dtype, np.integer)
    try:
        return arr.dtype == np.dtype(key)  # e.g. 'float64', 'int32'
    except TypeError:
        return True  # unknown key → do not fail hard

def _shape_matches(arr: np.ndarray, shape: tuple[int | None, ...]) -> bool:
    if arr.ndim != len(shape):
        return False
    return all(dim is None or dim == s for s, dim in zip(arr.shape, shape))

# ---------- main checker ----------
def check_type(value, hint, cls, obj=None) -> bool:
    """
    Best-effort runtime checker tolerant of *string* annotations (no eval).
    Supports:
      - typing objects: list[T], set[T], dict[K,V], tuple[...,], Union/|, Annotated
      - string 'Annotated[NDArray[float], (shape,)]' (dtype+shape)
      - plain string class names (accept subclasses via MRO)
      - string unions 'A | B'
    """
    # -------- STRING annotations path (no resolution) --------
    if isinstance(hint, str):
        h = hint.strip()

        # Handle plain "NDArray[...]" (dtype-only) without Annotated
        if _is_ndarray_base_str(h):
            if not isinstance(value, np.ndarray):
                return False
            dtype_key = _extract_ndarray_dtype_key_from_str(h)
            return _dtype_matches(value, dtype_key)

        # String Annotated[...]
        parsed = _parse_annotated_str(h)
        if parsed:
            base_str, meta = parsed

            # NDArray with shape metadata
            if _is_ndarray_base_str(base_str) and meta:
                shape = _shape_tuple_from_str(meta[0])
                dtype_key = _extract_ndarray_dtype_key_from_str(base_str)
                if not isinstance(value, np.ndarray):
                    return False
                if shape is not None and not _shape_matches(value, shape):
                    return False
                return _dtype_matches(value, dtype_key)

            # Otherwise treat base as class-like name → accept subclasses via MRO
            return _mro_name_match(value, base_str)

        # String union: "A | B"
        if '|' in h:
            parts = _split_top_level(h, sep='|')
            return any(check_type(value, p.strip(), cls) for p in parts)

        # Simple string container: "list[str]" (lightweight support)
        if h.startswith('list[') and h.endswith(']'):
            inner = _name_from_str(h[5:-1])
            if not isinstance(value, list):
                return False
            if inner == 'str':
                return all(isinstance(x, str) for x in value)
            if inner in ('float', 'float32', 'float64'):
                return all(isinstance(x, (float, int)) for x in value)
            return True  # permissive other inners

        # Plain forward-ref name → subclass-friendly check
        return _mro_name_match(value, h)

    # -------- Structured typing objects path --------
    origin = get_origin(hint)

    # Annotated[T, meta...] (real object)
    if origin is Annotated:
        base, *meta = get_args(hint)
        if isinstance(value, np.ndarray) and meta and isinstance(meta[0], tuple):
            expected_shape = meta[0]
            base_args = get_args(base)  # e.g., NDArray[dtype]
            dtype_key = None
            if base_args:
                dtype_arg = base_args[0]
                if dtype_arg is float:
                    dtype_key = 'float'
                elif hasattr(dtype_arg, 'name'):  # np.float64
                    dtype_key = dtype_arg.name
            expected_shape_list = list(expected_shape)
            for i, item in enumerate(expected_shape):
                if type(item) == str:
                    expected_shape_list[i] = getattr(obj, item)
            expected_shape = tuple(expected_shape_list)
            return _shape_matches(value, expected_shape) and _dtype_matches(value, dtype_key)
        return check_type(value, base, cls)

    # NDArray[...] without shape meta
    if origin is np.ndarray:
        return isinstance(value, np.ndarray)

    # Builtins / classes
    if origin is None:
        try:
            return isinstance(value, hint)
        except TypeError:
            return True

    # list[T]
    if origin is list:
        (t,) = get_args(hint)
        return isinstance(value, list) and all(check_type(x, t, cls) for x in value)

    # set[T]
    if origin is set:
        (t,) = get_args(hint)
        return isinstance(value, set) and all(check_type(x, t, cls) for x in value)

    # dict[K, V]
    if origin is dict:
        kt, vt = get_args(hint)
        return (isinstance(value, dict)
                and all(check_type(k, kt, cls) and check_type(v, vt, cls) for k, v in value.items()))

    # tuple[T1, T2] or tuple[T, ...]
    if origin is tuple:
        args = get_args(hint)
        if len(args) == 2 and args[1] is Ellipsis:
            return isinstance(value, tuple) and all(check_type(x, args[0], cls) for x in value)
        return isinstance(value, tuple) and len(value) == len(args) and all(check_type(x, t, cls) for x, t in zip(value, args))

    # Union[...] (incl Optional[T])
    if origin is Union:
        return any(check_type(value, t, cls) for t in get_args(hint))

    # Fallback: ABCs (Iterable, Sequence, etc.)
    try:
        return isinstance(value, origin)
    except TypeError:
        return True
