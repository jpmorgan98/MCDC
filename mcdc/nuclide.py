import h5py
import numpy as np
import os

####

from mcdc.reaction import (
    ReactionNeutronCapture,
    ReactionNeutronElasticScattering,
    ReactionNeutronFission,
    decode_type,
)
from mcdc.objects import ObjectNonSingleton
from mcdc.prints import print_1d_array

# ======================================================================================
# Nuclide class
# ======================================================================================


class Nuclide(ObjectNonSingleton):
    """
    Nuclide represents a neutron-transport nuclide loaded from an HDF5 neutron reaction
    data library.

    The class reads nuclide metadata and reaction cross sections from an HDF5 file
    located under the directory specified by the ``MCDC_XSLIB`` environment variable.
    It constructs reaction objects for supported neutron reactions and accumulates
    the total macroscopic cross section on the provided energy grid.

    Parameters
    ----------
    nuclide_name : str
        Base filename (without extension) of the nuclide HDF5 file to load,
        e.g. ``"U235"`` expects a file ``U235.h5`` in ``$MCDC_XSLIB``.

    Attributes
    ----------
    ID : int
        Internal identifier inherited from :class:`ObjectNonSingleton`.
    name : str
        Human-readable nuclide name loaded from the file (dataset ``nuclide_name``).
    atomic_weight_ratio : float
        Atomic weight ratio (to neutron mass), from dataset ``atomic_weight_ratio``.
    reactions : list[ReactionBase]
        Instantiated reaction objects (capture, elastic_scattering, and optionally
        fission), each created via ``ReactionClass.from_h5_group(...)``.
    fissionable : bool
        True if the HDF5 contains a ``fission`` reaction; False otherwise.
    xs_energy_grid : numpy.ndarray
        1D energy grid (in eV) used by all reaction cross sections,
        from ``neutron_reactions/xs_energy_grid``.
    total_xs : numpy.ndarray
        Elementwise sum of cross sections (same shape as ``xs_energy_grid``) over all
        loaded reactions.

    Environment
    -----------
    MCDC_XSLIB : str
        Filesystem path to the directory containing the nuclide HDF5 files.

    Notes
    -----
    - Supported reaction group names under ``/neutron_reactions`` are:
      ``capture``, ``elastic_scattering``, and ``fission`` (if present).
    - The class assumes each reaction group provides an ``xs`` array on the same
      ``xs_energy_grid``; these are summed into ``total_xs``.
    - If you add new reaction types, extend the mapping in ``__init__`` from
      group name to the corresponding reaction class.

    Raises
    ------
    EnvironmentError
        If the ``MCDC_XSLIB`` environment variable is not set.
    FileNotFoundError
        If the HDF5 file ``{nuclide_name}.h5`` is not found in ``$MCDC_XSLIB``.
    KeyError
        If expected datasets/groups are missing from the HDF5 file.
    OSError
        If the HDF5 file cannot be opened (e.g., permissions/corruption).

    See Also
    --------
    ReactionBase : Abstract base class for all reactions.
    ReactionNeutronCapture : Capture (n,γ) reaction.
    ReactionNeutronElasticScattering : Elastic scattering (n,n).
    ReactionNeutronFission : Neutron-induced fission.

    Examples
    --------
    >>> n = Nuclide("U235")
    >>> n.fissionable
    True
    >>> n.total_xs.shape == n.xs_energy_grid.shape
    True
    """
    def __init__(self, nuclide_name):
        """
        Initialize and load nuclide data from the HDF5 library.

        This constructor:
          1. Resolves ``$MCDC_XSLIB/{nuclide_name}.h5``.
          2. Loads basic nuclide data and the shared energy grid.
          3. Instantiates supported reaction objects from their HDF5 groups.
          4. Accumulates ``total_xs`` as the sum of per-reaction cross sections.

        Parameters
        ----------
        nuclide_name : str
            Base name of the nuclide file (without ``.h5``).

        Raises
        ------
        EnvironmentError
            If ``MCDC_XSLIB`` is unset.
        FileNotFoundError
            If the file cannot be located.
        KeyError
            If required datasets/groups are missing.
        OSError
            If the file cannot be opened or read.
        """
        label = "nuclide"
        super().__init__(label)

        # Set attributes from the hdf5 file
        dir_name = os.getenv("MCDC_XSLIB")
        file_name = f"{nuclide_name}.h5"
        with h5py.File(f"{dir_name}/{file_name}", "r") as f:
            self.name = f["nuclide_name"][()].decode()
            self.atomic_weight_ratio = f["atomic_weight_ratio"][()]
            self.xs_energy_grid = f["neutron_reactions/xs_energy_grid"][()]

            self.fissionable = False
            self.reactions = []
            self.total_xs = np.zeros_like(self.xs_energy_grid)

            for reaction_type in f["neutron_reactions"]:
                if reaction_type == "xs_energy_grid":
                    continue

                if reaction_type == "capture":
                    ReactionClass = ReactionNeutronCapture

                elif reaction_type == "elastic_scattering":
                    ReactionClass = ReactionNeutronElasticScattering

                elif reaction_type == "fission":
                    self.fissionable = True
                    ReactionClass = ReactionNeutronFission

                reaction = ReactionClass.from_h5_group(f[f"neutron_reactions/{reaction_type}"])
                self.reactions.append(reaction)

                # Accumulate total XS
                self.total_xs += reaction.xs

    def __repr__(self):
        """
        Return a human-readable multi-line summary of the nuclide.

        Returns
        -------
        str
            A formatted string including ID, name, atomic weight ratio,
            the energy grid summary, and the list of reaction types.
        """
        text = "\n"
        text += f"Nuclide\n"
        text += f"  - ID: {self.ID}\n"
        text += f"  - Name: {self.name}\n"
        text += f"  - Atomic weight ratio: {self.atomic_weight_ratio}\n"
        text += f"  - XS energy grid {print_1d_array(self.xs_energy_grid)} eV\n"
        text += f"  - Reactions\n"
        for reaction in self.reactions:
            text += f"    - {decode_type(reaction.type)}\n"
        return text
