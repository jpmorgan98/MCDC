import numpy as np

####

from mcdc.object_.data import DataPolynomial, DataTable
from mcdc.object_.distribution import DistributionMaxwellian, DistributionMultiPDF
from mcdc.object_.base import (
    ObjectNonSingleton,
    ObjectPolymorphic,
    ObjectSingleton,
)
from mcdc.object_.reaction import ReactionNeutronFission
from mcdc.util import flatten

# ======================================================================================
# Python object to Numba structured data converter
# ======================================================================================

# Supported base types
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


def numbafy_object(object_, structures, records, data):
    structure = []
    record = ()

    # Set ID with numba ID
    if isinstance(object_, ObjectNonSingleton):
        structure.append(("ID", type(object_.numba_ID)))
        record += (object_.numba_ID,)

    # List supported attributes of the object
    attribute_names = [
        x
        for x in dir(object_)
        if (
            x[:2] != "__"
            and not callable(getattr(object_, x))
            and x not in object_.non_numba
        )
    ]
    
    # Loop over the supported attributes
    for attribute_name in attribute_names:
        attribute = getattr(object_, attribute_name)

        # Convert list of supported types into Numpy array
        if type(attribute) == list:
            if len(attribute) == 0 or type(attribute[0]) in type_map.keys():
                attribute = np.array(attribute)

        # Scalar
        if type(attribute) in type_map.keys():
            structure.append((attribute_name, type_map[type(attribute)]))
            record += (attribute,)

        # Numpy array
        elif type(attribute) == np.ndarray:
            structure.append((f"{attribute_name}_offset", "i8"))
            structure.append((f"{attribute_name}_length", "i8"))

            offset = len(data)
            length = len(attribute.flatten())
            record += (offset, length)

            data.extend(attribute.flatten())

        # Polymorphic object
        elif isinstance(attribute, ObjectPolymorphic):
            structure.append((f"{attribute_name}_type", "i8"))
            structure.append((f"{attribute_name}_ID", "i8"))
            record += (attribute.type, attribute.numba_ID)

        # Non-polymorphic object
        elif isinstance(attribute, ObjectNonSingleton):
            structure.append((f"{attribute_name}_ID", "i8"))
            record += (attribute.numba_ID,)

        # List of Non-singleton objects
        elif type(attribute) == list:
            # Flatten the list
            attribute_flatten = list(flatten(attribute))

            if not isinstance(attribute_flatten[0], ObjectNonSingleton):
                print(
                    f"[ERROR] Get a list of non-object for {attribute_name}: {attribute}"
                )
                exit()

            # List of non-polymorphic objects
            if not isinstance(attribute_flatten[0], ObjectPolymorphic):
                structure.append((f"N_{attribute_name[:-1]}", "i8"))
                structure.append((f"{attribute_name[:-1]}_IDs_offset", "i8"))

                length = len(attribute_flatten)
                offset = len(data)
                record += (length, offset)

                data.extend([-1] * length)
                for i, subobject in enumerate(attribute_flatten):
                    data[offset + i] = subobject.numba_ID

            # List of polymorphic objects
            else:
                structure.append((f"N_{attribute_name[:-1]}", "i8"))
                structure.append((f"{attribute_name[:-1]}_types_offset", "i8"))
                structure.append((f"{attribute_name[:-1]}_IDs_offset", "i8"))

                length = len(attribute_flatten)
                offset_type = len(data)
                offset_id = offset_type + length
                record += (length, offset_type, offset_id)

                data.extend([-1] * length * 2)
                for i, subobject in enumerate(attribute_flatten):
                    # Generate the numba object
                    data[offset_type + i] = subobject.type
                    data[offset_id + i] = subobject.numba_ID

        # Dictionary
        else:
            print(f"[ERROR] Unsupported attribute: {attribute_name}: {attribute}")
            exit()

    # Register the numbafied object
    if isinstance(object_, ObjectSingleton):
        structures[object_.label] = np.dtype(structure)
        records[object_.label] = record
    elif isinstance(object_, ObjectNonSingleton):
        structures[object_.label] = np.dtype(structure)
        records[object_.label].append(record)


def generate_numba_objects(simulation):
    # Create necessary dummies
    if not simulation.settings.multigroup_mode:
        vector = np.zeros(1)
        if not any([isinstance(x, DataPolynomial) for x in simulation.data]):
            polynomial = DataPolynomial(vector)
            data_1d = polynomial
        if not any([isinstance(x, DataTable) for x in simulation.data]):
            table = DataTable(vector, vector)
            data_1d = table
        if not any([isinstance(x, DistributionMultiPDF) for x in simulation.distributions]):
            multipdf = DistributionMultiPDF(vector, vector, vector, vector)
            distribution = multipdf
        if not any([isinstance(x, DistributionMaxwellian) for x in simulation.distributions]):
            maxwellian = DistributionMaxwellian(0.0, vector, vector)
            distribution = maxwellian
        if not any([isinstance(x, ReactionNeutronFission) for x in simulation.reactions]):
            ReactionNeutronFission(vector, data_1d, distribution, [data_1d], [distribution], vector)

    object_list = [
        # Physics
        simulation.data,
        simulation.distributions,
        simulation.materials,
        simulation.nuclides,
        simulation.reactions,
        
        # Geometry
        simulation.cells,
        simulation.lattices,
        simulation.surfaces,
        simulation.universes,
        simulation.meshes,
        [simulation.settings],
        simulation.tallies,
    ]

    # Flatten the objects
    object_list = list(flatten(object_list))

    # Containers for structures and records for all object types
    structures = {}
    records = {}
    for object_ in object_list:
        structures[object_.label] = []
        if isinstance(object_, ObjectNonSingleton):
            records[object_.label] = []
    data = []

    # Loop over all objects
    for object_ in object_list:
        numbafy_object(object_, structures, records, data)

    data = np.array(data, dtype=np.float64)

    return structures, records, data
