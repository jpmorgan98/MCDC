# ======================================================================================
# Object base classes
# ======================================================================================


class ObjectBase:
    def __init__(self, label, register):
        self.label = label
        self.numbafied = False
        self.non_numba = ["non_numba", "label", "numbafied"]

        if register and isinstance(self, ObjectNonSingleton):
            register_object(self)


class ObjectSingleton(ObjectBase):
    def __init__(self, label):
        super().__init__(label, register=False)


class ObjectNonSingleton(ObjectBase):
    def __init__(self, label, register=True):
        self.ID = -1
        self.numba_ID = -1
        super().__init__(label, register)
        self.non_numba += ["ID", "numba_ID"]


class ObjectPolymorphic(ObjectNonSingleton):
    def __init__(self, label, type_, register=True):
        self.type = type_
        super().__init__(label, register)
        self.non_numba += ["type"]


class ObjectOverriding(ObjectPolymorphic):
    def __init__(self, label, type_, register=True):
        super().__init__(label, type_, register)


# ======================================================================================
# Helper functions
# ======================================================================================


def register_object(object_):
    from mcdc.object_.simulation import simulation

    from mcdc.object_.cell import Region, Cell, Universe, Lattice
    from mcdc.object_.data import DataBase
    from mcdc.object_.distribution import DistributionBase
    from mcdc.object_.material import MaterialBase
    from mcdc.object_.mesh import MeshBase
    from mcdc.object_.nuclide import Nuclide
    from mcdc.object_.reaction import ReactionBase
    from mcdc.object_.surface import Surface
    from mcdc.object_.tally import TallyBase

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
    elif isinstance(object_, Surface):
        object_list = simulation.surfaces
    elif isinstance(object_, TallyBase):
        object_list = simulation.tallies
    elif isinstance(object_, Universe):
        object_list = simulation.universes

    object_.ID = len(object_list)
    if not isinstance(object_, ObjectPolymorphic):
        object_.numba_ID = len(object_list)
    else:
        object_.numba_ID = sum([x.type == object_.type for x in object_list])
    object_list.append(object_)
