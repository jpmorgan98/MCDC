from __future__ import annotations

####

import numpy as np

from pathlib import Path

####

import mcdc
import mcdc.object_ as object_module
import mcdc.object_.base as base

from mcdc.object_.base import ObjectBase, ObjectNonSingleton, ObjectPolymorphic, ObjectSingleton
from mcdc.object_.particle import Particle, ParticleBank, ParticleData
from mcdc.print_ import print_error
from mcdc.util import flatten

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

bank_names = ['bank_active', 'bank_census', 'bank_source', 'bank_future']

# ======================================================================================
# Get MC/DC classes
# ======================================================================================

all_classes = [ParticleData, Particle]
mcdc_classes = [ParticleData, Particle]
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

def generate_numba_objects(simulation):
    # ==================================================================================
    # Allocate Python annotations and Numba structures, records, and data for the classes
    # ==================================================================================

    annotations = {}
    structures = {}
    records = {}
    data = []

    for mcdc_class in mcdc_classes:
        annotations[mcdc_class.label] = {}
        structures[mcdc_class.label] = []
        if issubclass(mcdc_class, ObjectNonSingleton):
            records[mcdc_class.label] = []
        else:
            records[mcdc_class.label] = {}

    # Particle banks
    for name in bank_names:
        annotations[name] = {}
        structures[name] = []

    # Move simulation to last
    annotations['simulation'] = annotations.pop('simulation')
    structures['simulation'] = structures.pop('simulation')
    records['simulation'] = records.pop('simulation')
    
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
    
    # Particle banks
    for name in bank_names:
        annotations[name] = {
            k: v
            for k, v in ParticleBank.__annotations__.items()
            if k not in ["label", "non_numba"]
            and (
                "non_numba" not in dir(ParticleBank)
                or (
                    "non_numba" in dir(ParticleBank)
                    and k not in ParticleBank.non_numba
                )
            )
        }

    # ==================================================================================
    # Set the structures based on the annotations
    # ==================================================================================

    # Temporary simulation object structure
    simulation_object_structure = []
    for field in annotations['simulation']:
        hint = annotations['simulation'][field]
        hint_origin = get_origin(hint)
        hint_args = get_args(hint)
       
        if hint in all_classes:
            simulation_object_structure.append((field, hint))
            continue
        if hint_origin == list and hint_args[0] in all_classes:
            simulation_object_structure.append((field, list, hint_args[0]))
            continue

    for label in annotations.keys():
        set_structure(label, structures, annotations)
    
    # Add particles to particle banks and add particle banks to the simulation
    for name in bank_names:
        bank = getattr(simulation, name)
        size = int(bank.size[0])
        structures[name] += [('particles', into_dtype(structures['particle_data']), (size,))]
        #
        structures['simulation'] = [(name, into_dtype(structures[name]))] + structures['simulation']

    # ==================================================================================
    # Set records and data based on the simulation structures and objects
    # ==================================================================================

    # Allocate object containers
    objects = {}
    for mcdc_class in mcdc_classes:
        if issubclass(mcdc_class, ObjectNonSingleton):
            objects[mcdc_class] = []
        else:
            objects[mcdc_class] = None

    # Gather the objects from the simulation
    attribute_names = [
        x
        for x in dir(simulation)
        if (
            not x.startswith("__")
            and not callable(getattr(simulation, x))
            and x not in simulation.non_numba
        )
    ]
    for attribute_name in attribute_names:
        attribute = getattr(simulation, attribute_name)
        if type(attribute) in mcdc_classes:
            objects[type(attribute)] = attribute
        if type(attribute) == list:
            for item in attribute:
                if type(item) in mcdc_classes:
                    objects[type(item)].append(item)

    # Set the objects
    for class_ in objects.keys():
        if issubclass(class_, ObjectNonSingleton):
            for object_ in objects[class_]:
                set_object(object_, structures, records, data)
        else:
            object_ = objects[class_]
            if object_ is not None:
                set_object(object_, structures, records, data)

    # ==================================================================================
    # Finalize the simulation object structure and set record
    # ==================================================================================

    new_structure = []
    record = records['simulation']
    for item in simulation_object_structure:
        field = item[0]
        type_1 = item[1]

        # List of objects
        if type_1 == list:
            type_2 = item[2]

            # List of non-polymorphics
            if item[2] not in polymorphic_bases:
                N = len(records[item[2].label])
                new_structure.append((field, into_dtype(structures[item[2].label]), (N,)))
                new_structure.append((f"N_{plural_to_singular(field)}", "i8"))
                record[f"N_{plural_to_singular(field)}"] = N

            # List of polymorphics
            else:
                for class_ in mcdc_classes:
                    if issubclass(class_, type_2):
                        N = len(records[class_.label])
                        new_structure.append((singular_to_plural(class_.label), into_dtype(structures[class_.label]), (N,)))
                        new_structure.append((f"N_{class_.label}", "i8"))
                        record[f"N_{class_.label}"] = N

        # Singleton
        elif item[1] in mcdc_classes and issubclass(item[1], ObjectSingleton):
            new_structure.append((field, into_dtype(structures[item[1].label])))

        else:
            print_error(f"Unknown type: {item}")

    structures['simulation'] = new_structure + structures['simulation']

    # GPU interop.
    structures['simulation'] += [
        ("gpu_state_pointer", 'u8'),
        ("source_program_pointer", 'u8'),
        ("precursor_program_pointer", 'u8'),
        ("source_seed", 'u8'),
    ]
    records['simulation']['gpu_state_pointer'] = 0
    records['simulation']['source_program_pointer'] = 0
    records['simulation']['precursor_program_pointer'] = 0
    records['simulation']['source_seed'] = 0

    # Set other record and data for simulation
    set_object(simulation, structures, records, data)

    # Print the fields
    with open(f"{Path(mcdc.__file__).parent}/object_/numba_types.py", "w") as f:
        text = "# The following is automatically generated by code_factory.py\n\n"
        text += "from mcdc.code_factory import into_dtype\n\n"

        for label in structures.keys():
            text += f"{label} = into_dtype([\n"
            structure = structures[label]
            for item in structure:
                if type(item[1]) != np.dtypes.VoidDType:
                    text += f"    {item},\n"
                else:
                    if len(item) == 3:
                        text += f"    ('{item[0]}', {plural_to_singular(item[0])}, {item[2]}),\n"
                    else:   
                        text += f"    ('{item[0]}', {item[0]}),\n"
            text += "])\n\n"

        f.write(text)

    # ==================================================================================
    # Set with records
    # ==================================================================================

    # The global structure/variable container
    mcdc_simulation_arr = np.zeros(1, dtype=into_dtype(structures['simulation']))
    mcdc_simulation = mcdc_simulation_arr[0]

    record = records['simulation']
    structure = structures['simulation']
    for item in structure:
        field = item[0]
        field_type = item[1]
        size = -1
        if len(item) == 3:
            size = item[2][0]

        # Skip particle banks
        if field in bank_names:
            continue

        # Simple attribute
        if type(field_type) != np.dtypes.VoidDType:
            mcdc_simulation[field] = record[field]

        # MC/DC objects
        else:
            # Singleton
            if size == -1:
                for sub_item in structures[field]:
                    mcdc_simulation[field][sub_item[0]] = records[field][sub_item[0]]
            # Non-singleton
            else:
                singular_field = plural_to_singular(field)
                for i in range(size):
                    for sub_item in structures[singular_field]:
                        mcdc_simulation[field][i][sub_item[0]] = records[singular_field][i][sub_item[0]]

    return mcdc_simulation_arr, data


def set_structure(label, structures, annotations):
    structure = structures[label]
    annotation = annotations[label]

    for field in annotation:
        hint = annotation[field]
        hint_decoded = decode_annotated_ndarray(hint)
        hint_origin = get_origin(hint)
        hint_args = get_args(hint)

        # Skip simulation object structure
        if label == 'simulation':
            if hint in all_classes:
                continue
            if hint_origin == list and hint_args[0] in all_classes:
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
           structure.append((field, type_map[type_], shape))

        # MC/DC classes
        elif non_polymorphic(hint):
            structure.append((f"{field}_ID", "i8"))
        elif polymorphic_base(hint):
            structure.append((f"{field}_type", "i8"))
            structure.append((f"{field}_ID", "i8"))

        # List of MC/DC classes
        elif list_of_non_polymorphics:
            singular = plural_to_singular(field)
            structure.append((f"N_{singular}", "i8"))
            structure.append((f"{singular}_IDs_offset", "i8"))
        elif list_of_polymorphic_bases:
            singular = plural_to_singular(field)
            structure.append((f"N_{singular}", "i8"))
            structure.append((f"{singular}_types_offset", "i8"))
            structure.append((f"{singular}_IDs_offset", "i8"))

        # Unknown type
        else:
            print_error(f"Unknown type hint for {label}/{field}: {hint}") 


def set_object(object_, structures, records, data):
    structure = structures[object_.label]
    record = {}

    if object_.label == 'simulation':
        record = records['simulation']

    # Straightforwardly set up attributes
    for key in [x[0] for x in structure]:
        if key in dir(object_):
            # Skip if set already
            if key in record.keys():
                continue
            record[key] = getattr(object_, key)

    # Loop over the supported attributes
    attribute_names = [
        x
        for x in dir(object_)
        if (
            x[:2] != "__"
            and not callable(getattr(object_, x))
        )
    ]
    if "non_numba" in dir(object_):
        attribute_names = list(set(attribute_names) - set(object_.non_numba))
    for attribute_name in attribute_names:
        # Skip if set already
        if attribute_name in record.keys():
            continue

        attribute = getattr(object_, attribute_name)

        # Convert list of supported types into Numpy array
        if type(attribute) == list:
            if len(attribute) == 0 or type(attribute[0]) in type_map.keys():
                attribute = np.array(attribute)

        # Numpy array
        if type(attribute) == np.ndarray:
            attribute_flatten = attribute.flatten()
            record[f"{attribute_name}_offset"] = len(data)
            record[f"{attribute_name}_length"] = len(attribute_flatten)
            data.extend(attribute_flatten)
        
        # Polymorphic object
        elif isinstance(attribute, ObjectPolymorphic):
            record[f"{attribute_name}_type"] = attribute.type
            record[f"{attribute_name}_ID"] = attribute.numba_ID

        # Non-polymorphic object
        elif isinstance(attribute, ObjectNonSingleton):
            record[f"{attribute_name}_ID"] = attribute.numba_ID
        
        # List of Non-singleton objects
        elif type(attribute) == list:
            # Flatten the list
            attribute_flatten = list(flatten(attribute))
            singular_name = plural_to_singular(attribute_name)

            if not isinstance(attribute_flatten[0], ObjectNonSingleton):
                print(
                    f"[ERROR] Get a list of non-object for {attribute_name}: {attribute}"
                )
                exit()

            # List of non-polymorphic objects
            if not isinstance(attribute_flatten[0], ObjectPolymorphic):
                record[f"N_{singular_name}"] = len(attribute_flatten)
                record[f"{singular_name}_IDs_offset"] = len(data)
                data.extend([x.numba_ID for x in attribute_flatten])

            # List of polymorphic objects
            else:
                length = len(attribute_flatten)
                offset_type = len(data)
                offset_id = offset_type + length

                record[f"N_{singular_name}"] = length
                record[f"{singular_name}_type_offset"] = offset_type
                record[f"{singular_name}_IDs_offset"] = offset_id
                data.extend([x.type for x in attribute_flatten])
                data.extend([x.numba_ID for x in attribute_flatten])

    # Complete for simulation object
    if object_.label == 'simulation':
        return

    # Check structure-record compatibility
    missing = set([x[0] for x in structure]) - set(record.keys())
    if len(missing) > 0:
        print_error(f"Missing structure keys in record for {object_.label}: {missing}")

    # Register the record
    if isinstance(object_, ObjectSingleton):
        records[object_.label] = record
    elif isinstance(object_, ObjectNonSingleton):
        records[object_.label].append(record)
        

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


# ======================================================================================
# Misc.
# ======================================================================================

def plural_to_singular(word: str) -> str:
    """
    Convert a plural English noun (possibly underscore-separated) to singular.
    Applies only to the last word and handles common irregulars.
    """
    irregulars = {
        'universes': 'universe',
        'children': 'child',
        'men': 'man',
        'women': 'woman',
        'people': 'person',
        'mice': 'mouse',
        'geese': 'goose',
        'teeth': 'tooth',
        'feet': 'foot',
        'indices': 'index',
        'matrices': 'matrix',
        'criteria': 'criterion',
        'data': 'data'  # invariant
    }

    parts = word.lower().split('_')
    w = parts[-1]

    if w in irregulars:
        parts[-1] = irregulars[w]
    elif w.endswith('ies') and len(w) > 3:
        parts[-1] = w[:-3] + 'y'
    elif w.endswith('ves') and len(w) > 3:
        parts[-1] = w[:-3] + 'f'
    elif w.endswith('oes'):
        parts[-1] = w[:-2]
    elif any(w.endswith(suffix) for suffix in ('ses', 'xes', 'zes', 'ches', 'shes')):
        parts[-1] = w[:-2]
    elif w.endswith('s') and not w.endswith('ss'):
        parts[-1] = w[:-1]

    return '_'.join(parts)


def singular_to_plural(word: str) -> str:
    """
    Convert a singular English noun (possibly underscore-separated) to plural.
    Applies only to the last word and handles common irregulars.
    """
    irregulars = {
        'universe': 'universes',
        'child': 'children',
        'man': 'men',
        'woman': 'women',
        'person': 'people',
        'mouse': 'mice',
        'goose': 'geese',
        'tooth': 'teeth',
        'foot': 'feet',
        'index': 'indices',
        'matrix': 'matrices',
        'criterion': 'criteria',
        'data': 'data'  # invariant
    }

    parts = word.lower().split('_')
    w = parts[-1]

    if w in irregulars:
        parts[-1] = irregulars[w]
    elif w.endswith('y') and w[-2:] not in ('ay', 'ey', 'iy', 'oy', 'uy'):
        parts[-1] = w[:-1] + 'ies'
    elif w.endswith('f'):
        parts[-1] = w[:-1] + 'ves'
    elif w.endswith('fe'):
        parts[-1] = w[:-2] + 'ves'
    elif w.endswith(('s', 'x', 'z', 'ch', 'sh')):
        parts[-1] = w + 'es'
    else:
        parts[-1] = w + 's'

    return '_'.join(parts)
