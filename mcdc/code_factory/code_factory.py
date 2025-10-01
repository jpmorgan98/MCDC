import mcdc.object_ as object_
import mcdc.object_.base as base

from mcdc.object_.base import ObjectBase, ObjectNonSingleton

base_classes = [
    getattr(base, x)
    for x in dir(base)
    if isinstance(getattr(base, x), type) and issubclass(getattr(base, x), ObjectBase)
]


def generate_numba_objects(simulation):
    # ==================================================================================
    # Get MC/DC classes
    # ==================================================================================

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
                and item_name[-4:] != "Base"
                and item not in mcdc_classes
            ):
                mcdc_classes.append(item)

    # ==================================================================================
    # Allocate Numba structures and records for the classes
    # ==================================================================================

    structures = {}
    records = {}
    for item in mcdc_classes:
        print(item, item.label)
        structures[item.label] = []
        if isinstance(item, ObjectNonSingleton):
            records[item.label] = []
    data = []
