"""
Object registry and base-class framework
=======================================

This module provides the **global object registry** and a small hierarchy of
base classes used across MCDC. All constructible entities (materials, nuclides,
reactions, data containers, geometry, settings) derive from these bases and
**auto-register** themselves on construction.

Overview
--------
- Each non-singleton object is appended to a dedicated module-level list and
  assigned a stable integer ``ID`` matching its insertion order.
- Singleton objects (e.g., :class:`mcdc.settings.Settings`) are not listed and
  do not receive an ``ID``; they typically assign themselves to a dedicated
  global (e.g., ``objects.settings``).
- The registry is centralized via :func:`register_object`, which is called by
  :class:`ObjectBase` during construction.

Global registries
-----------------
The following module-level containers are the canonical inventories used
throughout the codebase:

- ``materials`` : list
    Overriding–polymorphic materials
    (:class:`mcdc.material.Material`, :class:`mcdc.material.MaterialMG`).
- ``nuclides`` : list
    Non-singleton :class:`mcdc.nuclide.Nuclide`.
- ``reactions`` : list
    Polymorphic :class:`mcdc.reaction.ReactionBase` and subclasses.
- ``data_containers`` : list
    Polymorphic :class:`mcdc.data_container.DataContainer` and subclasses.
- ``surfaces`` : list
    Non-singleton :class:`mcdc.surface.Surface`.
- ``regions`` : list
    Non-singleton :class:`mcdc.cell.Region`.
- ``cells`` : list
    Non-singleton :class:`mcdc.cell.Cell`.
- ``settings`` : Optional[:class:`mcdc.settings.Settings`]
    Singleton settings instance.

Base-class hierarchy
--------------------
.. code-block:: text

    ObjectBase
    ├── ObjectSingleton
    └── ObjectNonSingleton
        └── ObjectPolymorphic (adds ``type`` code)
            └── ObjectOverriding  (used when higher-level types override behavior)

Key concepts
------------
Auto-registration
    ``ObjectBase.__init__`` calls :func:`register_object(self)`. The dispatcher
    routes the object to the correct registry based on its concrete type.

IDs for non-singletons
    Objects deriving from :class:`ObjectNonSingleton` receive ``ID`` equal to
    their index in the destination registry at insertion time. They also carry
    ``ID_numba`` for use by Numba backends.

Numba bookkeeping
    All bases maintain ``non_numba`` (a list of attribute names excluded from
    Numba-struct lowering) and a ``numbafied`` flag for transforms.

Thread-safety
    The registries are plain Python lists. If you construct objects from
    multiple threads, external synchronization is required to avoid ID races.

API reference
-------------
Classes
^^^^^^^
- :class:`ObjectBase`
    Root base; stores ``label``, tracks Numba state, auto-registers.
- :class:`ObjectSingleton`
    For singletons; not listed, no ``ID``.
- :class:`ObjectNonSingleton`
    Listed objects; receive ``ID`` and carry ``ID_numba``.
- :class:`ObjectPolymorphic`
    Non-singleton plus integer ``type`` code.
- :class:`ObjectOverriding`
    Polymorphic base used by types that override behavior (e.g., materials).

Functions
^^^^^^^^^
- :func:`register_object(object_)`
    Dispatches new instances into the appropriate registry and assigns IDs
    to non-singletons.

See Also
--------
mcdc.material.Material, mcdc.material.MaterialMG
    Concrete material types (CE vs. MG).
mcdc.nuclide.Nuclide
    CE nuclide loader (aggregates reaction data).
mcdc.reaction.ReactionBase and subclasses
    Reaction model hierarchy.
mcdc.data_container.DataContainer and subclasses
    Data parameterizations used by reactions.
mcdc.settings.Settings
    Global simulation settings (singleton).
"""


class ObjectBase:
    """
    Root base class for all MCDC objects.

    Responsibilities
    ----------------
    - Stores a human-readable ``label``.
    - Tracks Numba conversion state via ``numbafied`` and ``non_numba``.
    - **Auto-registers** itself by calling :func:`register_object` on construction.

    Parameters
    ----------
    label : str
        Short descriptive label for debugging/printing.

    Attributes
    ----------
    label : str
        Object label.
    numbafied : bool
        Whether this object has been transformed for Numba use.
    non_numba : list[str]
        Attribute names that should not be moved into Numba structs.

    Notes
    -----
    Every subclass constructor calls ``super().__init__(label)`` which triggers
    :func:`register_object` to place the instance into the appropriate global
    list in this module (e.g., ``materials``, ``nuclides``).

    See Also
    --------
    register_object : Dispatcher that inserts new objects into global registries.
    ObjectNonSingleton, ObjectPolymorphic, ObjectOverriding, ObjectSingleton
        Derived base types used throughout the codebase.
    """
    def __init__(self, label, automatic_registration=True):
        self.label = label
        self.numbafied = False
        self.non_numba = ["non_numba", "label", "numbafied"]
        if automatic_registration:
            register_object(self)


class ObjectSingleton(ObjectBase):
    """
    Base class for **singleton** objects (exactly one live instance expected).

    Parameters
    ----------
    label : str
        Descriptive label.

    Notes
    -----
    - Singletons are **not** assigned an ``ID`` and are not appended to a list
      by :func:`register_object`. Instead, each concrete singleton typically
      assigns itself to a dedicated global (e.g., :class:`mcdc.settings.Settings`
      sets ``mcdc.objects.settings = self`` in its ``__post_init__``).

    See Also
    --------
    mcdc.settings.Settings : Main global settings singleton.
    register_object : Called during construction.
    """
    def __init__(self, label):
        super().__init__(label)


class ObjectNonSingleton(ObjectBase):
    """
    Base class for objects that live in a **global list** and receive an ``ID``.

    Parameters
    ----------
    label : str
        Descriptive label.

    Attributes
    ----------
    ID : int
        Index into the corresponding global list (assigned at registration).
    ID_numba : int
        Numba-side ID (initialized to -1; assigned during Numba transforms).

    Notes
    -----
    - During construction, :func:`register_object` appends the instance to the
      appropriate module-level list and assigns ``ID = len(list_before_append)``.
    - ``ID_numba`` and ``ID`` are added to ``non_numba`` by default.

    See Also
    --------
    register_object : Dispatcher that determines the destination list.
    """
    def __init__(self, label, automatic_registration=True):
        self.ID = -1
        super().__init__(label, automatic_registration)
        self.ID_numba = -1
        self.non_numba += ["ID", "ID_numba"]


class ObjectPolymorphic(ObjectNonSingleton):
    """
    Base class for **polymorphic** non-singleton objects with a ``type`` code.

    Parameters
    ----------
    label : str
        Descriptive label.
    type_ : int
        Implementation/type discriminator used at runtime.

    Attributes
    ----------
    type : int
        Polymorphic type code. Added to ``non_numba``.

    See Also
    --------
    ObjectOverriding : Polymorphic subclass used when higher-level types override behavior.
    mcdc.reaction.ReactionBase : Example of a polymorphic hierarchy.
    mcdc.data_container.DataContainer : Another polymorphic family.
    """
    def __init__(self, label, type_, automatic_registration=True):
        super().__init__(label, automatic_registration)
        self.type = type_
        self.non_numba += ["type"]


class ObjectOverriding(ObjectPolymorphic):
    """
    Base class for **overriding** polymorphic objects.

    Parameters
    ----------
    label : str
        Descriptive label.
    type_ : int
        Implementation/type discriminator.

    Notes
    -----
    This is used where derived material types share an interface but override
    some behavior (e.g., :class:`mcdc.material.Material` vs.
    :class:`mcdc.material.MaterialMG`).

    See Also
    --------
    mcdc.material.Material, mcdc.material.MaterialMG
        Concrete overriding material types.
    """
    def __init__(self, label, type_, automatic_registration=True):
        super().__init__(label, type_, automatic_registration)


# The objects
materials = []          # Overriding-polymorphic (Material, MaterialMG)
nuclides = []           # Non-singleton (Nuclide)
reactions = []          # Polymorphic (ReactionBase and subclasses)
data_containers = []    # Polymorphic (DataContainer and subclasses)
surfaces = []           # Non-singleton (Surface)
regions = []            # Non-singleton (Region)
cells = []              # Non-singleton (Cell)
universes = [None]      # Non-singleton (Universe); None as root universe placeholder
lattices = []           # Non-singleton (Lattice)
settings = None         # Singleton (Settings)


# Helper functions
def register_object(object_):
    """
    Register a newly constructed object into the appropriate global registry.

    The dispatcher identifies the concrete class family (materials, nuclides,
    reactions, data containers, geometry elements) and appends **non-singleton**
    objects to the corresponding global list, assigning a stable ``ID`` that
    matches the insertion index.

    Parameters
    ----------
    object_ : ObjectBase
        Any object derived from :class:`ObjectBase`.

    Behavior
    --------
    - **Singletons** (instances of :class:`ObjectSingleton`) are *not* appended
      to lists and receive no ``ID`` here. Concrete singletons typically assign
      themselves to a dedicated global variable elsewhere (e.g., ``objects.settings``).
    - **Non-singletons** (instances of :class:`ObjectNonSingleton`) are appended
      to the correct list and get ``ID = len(list_before_append)``.

    See Also
    --------
    ObjectBase, ObjectSingleton, ObjectNonSingleton
        Base classes whose constructors trigger registration.
    mcdc.material.Material, mcdc.material.MaterialMG
        Materials (go to ``objects.materials``).
    mcdc.nuclide.Nuclide
        Nuclides (go to ``objects.nuclides``).
    mcdc.reaction.ReactionBase
        Reactions (go to ``objects.reactions``).
    mcdc.data_container.DataContainer
        Data containers (go to ``objects.data_containers``).
    mcdc.surface.Surface, mcdc.cell.Region, mcdc.cell.Cell
        Geometry elements (go to ``objects.surfaces``, ``objects.regions``, ``objects.cells``).
    """
    from mcdc.data_container import DataContainer
    from mcdc.material import MaterialBase
    from mcdc.nuclide import Nuclide
    from mcdc.reaction import ReactionBase
    from mcdc.surface import Surface
    from mcdc.cell import Region, Cell, Universe, Lattice

    global materials, nuclides, reactions, data_containers

    if isinstance(object_, MaterialBase):
        object_list = materials
    elif isinstance(object_, Nuclide):
        object_list = nuclides
    elif isinstance(object_, ReactionBase):
        object_list = reactions
    elif isinstance(object_, DataContainer):
        object_list = data_containers
    elif isinstance(object_, Surface):
        object_list = surfaces
    elif isinstance(object_, Region):
        object_list = regions
    elif isinstance(object_, Cell):
        object_list = cells
    elif isinstance(object_, Universe):
        object_list = universes
    elif isinstance(object_, Lattice):
        object_list = lattices

    if isinstance(object_, ObjectSingleton):
        object_list = object_
    if isinstance(object_, ObjectNonSingleton):
        object_.ID = len(object_list)
        object_list.append(object_)
