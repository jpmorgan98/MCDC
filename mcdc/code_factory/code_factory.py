from __future__ import annotations

####

import numpy as np

from pathlib import Path

####

import mcdc
import mcdc.object_ as object_module
import mcdc.object_.base as base

from mcdc.object_.base import ObjectBase, ObjectNonSingleton, ObjectPolymorphic, ObjectSingleton
from mcdc.print_ import print_error

base_classes = [
    getattr(base, x)
    for x in dir(base)
    if isinstance(getattr(base, x), type) and issubclass(getattr(base, x), ObjectBase)
]

type_map = {
    bool: "?",
    float: "f8",
    int: "i8",
    str: "U32",
    np.bool_: "?",
    np.float64: "f8",
    np.int64: "i8",
    np.uint64: "u8",
    np.str_: "U32",
}

# ======================================================================================
# Get MC/DC classes
# ======================================================================================

all_classes = []
mcdc_classes = []
polymorphic_bases = []

file_names = [x for x in dir(object_module) if x[:2] != "__" and x != "base"]
for file_name in file_names:
    file = getattr(object_module, file_name)
    item_names = dir(file)
    for item_name in item_names:
        item = getattr(file, item_name)
        if (
            isinstance(item, type)
            and issubclass(item, ObjectBase)
            and item not in all_classes
        ):
            all_classes.append(item)

            if (
                item not in base_classes
                and 'label' in dir(item)
                and item not in mcdc_classes
            ):
                mcdc_classes.append(item)

polymorphic_bases = [x for x in all_classes if x.__name__[-4:] == "Base"]

# ======================================================================================
# Numba object creation
# ======================================================================================

def set_structure(label, structures, annotations):
    structure = structures[label]
    annotation = annotations[label]

    for field in annotation:
        hint = annotation[field]
        hint_decoded = decode_annotated_ndarray(hint)
        hint_origin = get_origin(hint)
        hint_args = get_args(hint)

        # For simulation, skip lists of classes
        if label == 'simulation' and hint_origin == list and hint_args[0] in all_classes:
            continue

        # ==========================================================================
        # Get the type
        # ==========================================================================

        # Basics
        simple_scalar = hint in type_map.keys()
        simple_list = hint_origin == list and hint_args[0] in type_map.keys()
        numpy_array = hint_origin == np.ndarray
        fixed_size_array = hint_decoded is not None # Resolved later

        # MC/DC class
        non_polymorphic = lambda x: issubclass(x, ObjectNonSingleton) and x not in polymorphic_bases
        polymorphic_base = lambda x: x in polymorphic_bases 
        singleton = lambda x: issubclass(x, ObjectSingleton)

        # List of MC/DC classes
        list_of_non_polymorphics = hint_origin == list and non_polymorphic(hint_args[0])
        list_of_polymorphic_bases = hint_origin == list and polymorphic_base(hint_args[0])

        # ==========================================================================
        # Set the structure
        # ==========================================================================

        # Basics
        if simple_scalar:
            structure.append((field, type_map[hint]))
        elif simple_list or numpy_array:
            structure.append((f"{field}_offset", "i8"))
            structure.append((f"{field}_length", "i8"))
        elif fixed_size_array:
           shape = hint_decoded['shape']
           type_ = get_args(hint_decoded['dtype'])[0]
           structure.append((field, shape, type_))

        # MC/DC classes
        elif non_polymorphic(hint):
            structure.append((f"{field}_ID", "i8"))
        elif polymorphic_base(hint):
            structure.append((f"{field}_type", "i8"))
            structure.append((f"{field}_ID", "i8"))
        elif singleton(hint):
            # Set the singleton if not set yet
            if len(structures[field]) == 0:
                set_structure(field, structures, annotations)
            structure.append((f"{field}", into_dtype(structures[field])))

        # List of MC/DC classes
        elif list_of_non_polymorphics:
            structure.append((f"N_{field[:-1]}", "i8"))
            structure.append((f"{field[:-1]}_IDs_offset", "i8"))
        elif list_of_polymorphic_bases:
            structure.append((f"N_{field[:-1]}", "i8"))
            structure.append((f"{field[:-1]}_types_offset", "i8"))
            structure.append((f"{field[:-1]}_IDs_offset", "i8"))

        # Unknown type
        else:
            print_error(f"Unknown type hint for {label}/{field}: {hint}") 


def generate_numba_objects(simulation):
    # ==================================================================================
    # Allocate Python annotations and Numba structures, records, and data for the classes
    # ==================================================================================

    annotations = {}
    structures = {}
    records = {}
    data = []

    # Simulation-specific items
    object_list_names = []

    # Allocate with list
    for mcdc_class in mcdc_classes:
        annotations[mcdc_class.label] = {}
        structures[mcdc_class.label] = []
        if issubclass(mcdc_class, ObjectNonSingleton):
            records[mcdc_class.label] = []
    
    # ==================================================================================
    # Gather the annotations from the classes
    # ==================================================================================

    for mcdc_class in mcdc_classes:
        # Include all ancestors, but stop at the MC/DC base classes
        classes = []
        for item in mcdc_class.__mro__:
            if item in base_classes:
                break
            classes.append(item)

        # Get the annotations
        for class_ in classes:
            new_annotations = {
                k: v
                for k, v in class_.__annotations__.items()
                if k not in ["label", "non_numba"]
                and (
                    "non_numba" not in dir(class_)
                    or (
                        "non_numba" in dir(class_)
                        and k not in class_.non_numba
                    )
                )
            }
            # Evaluate stringified annotation
            if (
                len(new_annotations) > 0
                and type(next(iter(new_annotations.values()))) == str
            ):
                new_annotations = parse_annotations_dict(new_annotations)

            annotations[mcdc_class.label].update(new_annotations)

    # ==================================================================================
    # Set the structures based on the annotations
    # ==================================================================================

    for label in annotations.keys():
        set_structure(label, structures, annotations)

    # Print the fields
    with open(f"{Path(mcdc.__file__).parent}/object_/numba_fields.yaml", "w") as f:
        text = "# The following is automatically generated by code_factory.py\n\n"

        for label in structures.keys():
            text += f"{label}:\n"
            structure = structures[label]
            for item in structure:
                text += f"    - {item}\n"
            text += "\n"

        f.write(text)


# ======================================================================================
# Alignment Logic
# ======================================================================================
# While CPU execution can robustly handle all sorts of Numba types, GPU
# execution requires structs to follow some of the basic properties expected of
# C-style structs with standard layout:
#
#      - Every primitive field is aligned by its size, and padding is inserted
#        between fields to ensure alignment in arrays and nested data structures
#
#      - Every field has a unique address
#
# If these rules are violated, memory accesses made in GPUs may encounter
# problems. For example, in cases where an access is not at an address aligned
# by their size, a segfault or similar fault will occur, or information will be
# lost. These issues were fixed by providing a function, align, which ensures the
# field lists fed to np.dtype fulfill these requirements.
#
# The align function does the following:
#
#      - Tracks the cumulative offset of fields as they appear in the input list.
#
#      - Inserts additional padding fields to ensure that primitive fields are
#        aligned by their size
#
#      - Re-sizes arrays to have at least one element in their array (this ensure
#        they have a non-zero size, and hence cannot overlap base addresses with
#        other fields.
#


def fixup_dims(dim_tuple):
    return tuple([max(d, 1) for d in dim_tuple])


def align(field_list):
    result = []
    offset = 0
    pad_id = 0
    for field in field_list:
        if len(field) > 3:
            print_error(
                "Unexpected struct field specification. Specifications \
                        usually only consist of 3 or fewer members"
            )
        multiplier = 1
        if len(field) == 3:
            field = (field[0], field[1], fixup_dims(field[2]))
            for d in field[2]:
                multiplier *= d
        kind = np.dtype(field[1])
        size = kind.itemsize

        if kind.isbuiltin == 0:
            alignment = 8
        elif kind.isbuiltin == 1:
            alignment = size
        else:
            print_error("Unexpected field item type")

        size *= multiplier

        if offset % alignment != 0:
            pad_size = alignment - (offset % alignment)
            result.append((f"padding_{pad_id}", np.uint8, (pad_size,)))
            pad_id += 1
            offset += pad_size

        result.append(field)
        offset += size

    if offset % 8 != 0:
        pad_size = 8 - (offset % 8)
        result.append((f"padding_{pad_id}", np.uint8, (pad_size,)))
        pad_id += 1

    return result


def into_dtype(field_list):
    result = np.dtype(align(field_list), align=True)
    return result

# ======================================================================================
# Type parser
# ======================================================================================

from typing import Annotated, Any, ForwardRef, Optional, Union, get_args, get_origin
import numpy as np
from numpy.typing import NDArray


# --- Safe locals for eval + ForwardRef fallback ---
class _FwdRefDict(dict):
    """If a symbol isn't in the whitelist, treat it as a ForwardRef('Symbol')."""

    def __missing__(self, key):
        for class_ in all_classes:
            if key == class_.__name__:
                return class_
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


def decode_annotated_ndarray(hint):
    origin = get_origin(hint)
    if origin is Annotated:
        inner, metadata = get_args(hint)
        inner_origin = get_origin(inner)
        inner_args = get_args(inner)
        if inner_origin is np.ndarray:
            shape_type, dtype_type = inner_args
            return {
                "base": inner_origin,
                "shape": metadata,
                "shape_type": shape_type,
                "dtype": dtype_type,
            }
    return None
