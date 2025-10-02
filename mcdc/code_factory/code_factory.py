from __future__ import annotations

####

import mcdc.object_ as object_
import mcdc.object_.base as base

from mcdc.object_.base import ObjectBase, ObjectNonSingleton

base_classes = [
    getattr(base, x)
    for x in dir(base)
    if isinstance(getattr(base, x), type) and issubclass(getattr(base, x), ObjectBase)
]

# ======================================================================================
# Get MC/DC classes
# ======================================================================================

mcdc_classes = []

file_names = [x for x in dir(object_) if x[:2] != "__" and x != "base"]
for file_name in file_names:
    file = getattr(object_, file_name)
    item_names = dir(file)
    for item_name in item_names:
        item = getattr(file, item_name)
        if (
            isinstance(item, type)
            and issubclass(item, ObjectBase)
            and item not in base_classes
            and 'label' in dir(item)
            and item not in mcdc_classes
        ):
            mcdc_classes.append(item)


def generate_numba_objects(simulation):
    # ==================================================================================
    # Allocate Numba structures, records, and data for the classes
    # ==================================================================================

    structures = {}
    records = {}
    for mcdc_class in mcdc_classes:
        structures[mcdc_class.label] = []
        if isinstance(mcdc_class, ObjectNonSingleton):
            records[mcdc_class.label] = []
    data = []

    # ==================================================================================
    # Set the structures
    # ==================================================================================

    for mcdc_class in mcdc_classes:
        # Include all ancestors, but stop at the MC/DC base classes
        classes = []
        for item in mcdc_class.__mro__:
            if item in base_classes:
                break
            classes.append(item)

        # Get the annotations
        annotations = {}
        for class_ in classes:
            new_annotations = {
                k: v
                for k, v in class_.__annotations__.items()
                if "non_numba" in dir(class_)
                and k not in class_.non_numba + ["label", "non_numba"]
            }

            # Evaluate stringified annotation
            if (
                len(new_annotations) > 0
                and type(next(iter(new_annotations.values()))) == str
            ):
                new_annotations = parse_annotations_dict(new_annotations)

            annotations.update(new_annotations)

        # Set the structure based on the annotations
        for key in annotations.keys():
            structures[mcdc_class.label] += [(key, annotations[key])]


# ======================================================================================
# Type parser
# ======================================================================================

from typing import Annotated, Any, ForwardRef, Optional, Union
import numpy as np
from numpy.typing import NDArray


# --- Safe locals for eval + ForwardRef fallback ---
class _FwdRefDict(dict):
    """If a symbol isn't in the whitelist, treat it as a ForwardRef('Symbol')."""

    def __missing__(self, key):
        for mcdc_class in mcdc_classes:
            if key == mcdc_class.__name__:
                return mcdc_class
        return ForwardRef(key)


_SAFE_GLOBALS = {"__builtins__": {}}  # no builtins
_SAFE_LOCALS = _FwdRefDict(
    {
        # builtins
        "str": str,
        "int": int,
        "float": float,
        "bool": bool,
        "bytes": bytes,
        "object": object,
        "list": list,
        "dict": dict,
        "tuple": tuple,
        "set": set,
        # typing
        "Any": Any,
        "Annotated": Annotated,
        "Union": Union,
        "Optional": Optional,
        # numpy typing
        "NDArray": NDArray,
        # numpy dtypes (extend if you need more)
        "float64": np.float64,
        "float32": np.float32,
        "int64": np.int64,
        "int32": np.int32,
    }
)


def parse_type_hint_str(s: str):
    """
    Parse a stringified type hint into a runtime type/typing object.
    Unknown identifiers become ForwardRef('Name') so we don't import/resolve.
    """
    s = s.strip()
    # Special-case empty or 'None' if you ever pass those
    if s in {"None", "NoneType"}:
        return type(None)
    return eval(s, _SAFE_GLOBALS, _SAFE_LOCALS)


def parse_annotations_dict(ann: dict[str, str]) -> dict[str, object]:
    return {k: parse_type_hint_str(v) for k, v in ann.items()}
