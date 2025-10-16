from mcdc.object_.util import check_type
from mcdc.print_ import print_error


# ======================================================================================
# Object base classes
# ======================================================================================


class ObjectBase:
    def __init__(self, register):
        if register and isinstance(self, ObjectNonSingleton):
            register_object(self)
    
        if "non_numba" in dir(self):
            self.non_numba += ["non_numba", "label"]
        else:
            self.non_numba = ["non_numba", "label"]

    def __setattr__(self, key, value):
        hints = getattr(self.__class__, "__annotations__", {})
        if key in hints and not check_type(value, hints[key], self.__class__, self):
            print_error(f"{key} must be {hints[key]!r}, got {value!r}")
        super().__setattr__(key, value)


class ObjectSingleton(ObjectBase):
    def __init__(self):
        super().__init__(register=False)


class ObjectNonSingleton(ObjectBase):
    ID: int
    numba_ID: int
    def __init__(self, register=True):
        self.ID = -1
        self.numba_ID = -1
        super().__init__(register)

        if "non_numba" in dir(self):
            self.non_numba += ["ID", "numba_ID"]
        else:
            self.non_numba = ["ID", "numba_ID"]

class ObjectPolymorphic(ObjectNonSingleton):
    type: int
    def __init__(self, type_, register=True):
        self.type = type_
        super().__init__(register)


# ======================================================================================
# Helper functions
# ======================================================================================


def register_object(object_):
    from mcdc.object_.simulation import simulation

    from mcdc.object_.cell import Region, Cell
    from mcdc.object_.universe import Universe, Lattice
    from mcdc.object_.data import DataBase
    from mcdc.object_.distribution import DistributionBase
    from mcdc.object_.material import MaterialBase
    from mcdc.object_.mesh import MeshBase
    from mcdc.object_.nuclide import Nuclide
    from mcdc.object_.reaction import ReactionBase
    from mcdc.object_.source import Source
    from mcdc.object_.surface import Surface
    from mcdc.object_.tally import TallyBase

    object_list = []
    if isinstance(object_, Cell):
        object_list = simulation.cells
    elif isinstance(object_, DataBase):
        object_list = simulation.data
    elif isinstance(object_, DistributionBase):
        object_list = simulation.distributions
    elif isinstance(object_, Lattice):
        object_list = simulation.lattices
    elif isinstance(object_, MaterialBase):
        object_list = simulation.materials
    elif isinstance(object_, MeshBase):
        object_list = simulation.meshes
    elif isinstance(object_, Nuclide):
        object_list = simulation.nuclides
    elif isinstance(object_, ReactionBase):
        object_list = simulation.reactions
    elif isinstance(object_, Region):
        object_list = simulation.regions
    elif isinstance(object_, Source):
        object_list = simulation.sources
    elif isinstance(object_, Surface):
        object_list = simulation.surfaces
    elif isinstance(object_, TallyBase):
        object_list = simulation.tallies
    elif isinstance(object_, Universe):
        object_list = simulation.universes
    else:
        print_error(f"Unidentified object list for object {object_}")

    object_.ID = len(object_list)
    object_.numba_ID = len(object_list)
    if isinstance(object_, ObjectPolymorphic):
        object_.numba_ID = sum([x.type == object_.type for x in object_list])
    object_list.append(object_)
