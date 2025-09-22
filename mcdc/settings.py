import h5py
import numpy as np

from dataclasses import dataclass, field
from numpy.typing import NDArray

####

import mcdc.objects as objects

from mcdc.constant import *
from mcdc.material import MaterialMG
from mcdc.objects import ObjectSingleton
from mcdc.prints import print_error
from mcdc.util import is_sorted

# ======================================================================================
# Settings class
# ======================================================================================


@dataclass
class Settings(ObjectSingleton):
    """
    Global input settings for an MCDC simulation.

    This singleton aggregates run controls (particle counts, batches, RNG seed),
    simulation modes (continuous-energy vs. multigroup, eigenvalue mode),
    optional time-census boundaries, I/O options, and GPU targeting flags.

    Attributes
    ----------
    N_particle : int
        Number of source particles per batch (or total if single batch).
    N_batch : int
        Number of batches (default 1).
    rng_seed : int
        Random number generator seed.
    multigroup_mode : bool
        True if at least one material is :class:`MaterialMG`; set in ``__post_init__``.
    eigenvalue_mode : bool
        True after calling :meth:`set_eigenmode`.
    N_inactive : int
        Inactive (discarded) eigenvalue cycles.
    N_active : int
        Active eigenvalue cycles used for k statistics.
    k_init : float
        Initial guess for k in eigenvalue mode.
    use_gyration_radius : bool
        Whether to enable gyration radius constraint/measurement.
    gyration_radius_type : int
        One of ``GYRATION_RADIUS_*`` constants.
    use_source_file : bool
        If True, read initial particles from an HDF5 source file.
    source_file_name : str
        Path to the HDF5 source file (when enabled).
    time_boundary : float
        Global maximum time (default ``np.inf``).
    output_name : str
        Base name for output files.
    use_progress_bar : bool
        Enable progress reporting.
    save_input_deck : bool
        Persist a copy of the input deck.
    N_census : int
        Number of time-census boundaries (including the terminal ``inf``).
    census_time : numpy.ndarray
        Monotone array of census boundaries; last element is ``np.inf``.
    use_census_based_tally : bool
        If True, tallies use uniform meshes within each census interval.
    census_tally_frequency : int
        Number of equal sub-bins per census interval (if enabled).
    save_particle : bool
        If True, write final particle bank(s).
    active_bank_buffer : int
        Buffer size for active particle banks.
    census_bank_buffer_ratio, source_bank_buffer_ratio, future_bank_buffer_ratio : float
        Relative buffer multipliers for respective banks.
    target_gpu : bool
        If True, target GPU execution.

    Notes
    -----
    - ``multigroup_mode`` is determined automatically in ``__post_init__`` by
      checking whether the first material in ``mcdc.objects.materials`` is a
      :class:`MaterialMG`. Ensure materials are created before instantiating
      :class:`Settings`.
    - This class is a singleton via :class:`ObjectSingleton`; the last instance
      registers itself to ``mcdc.objects.settings``.

    See Also
    --------
    Material, MaterialMG
        Material definitions (continuous-energy vs. multigroup).
    Nuclide
        Continuous-energy nuclide loader used by :class:`Material`.
    ReactionBase, ReactionNeutronCapture, ReactionNeutronElasticScattering, ReactionNeutronFission
        Reaction classes aggregated by :class:`Nuclide`.
    DataTable, DataPolynomial, DataMultiPDF, DataMaxwellian
        Data containers used by reaction models.

    Examples
    --------
    >>> import mcdc
    >>>
    >>> # Define a simple material (1 atom/barn-cm of U-235)
    >>> mat = Material("fuel", {"U235": 1.0})
    >>>
    >>> # Create and register settings
    >>> s = Settings(N_particle=10000, N_batch=5, rng_seed=42)
    >>> s.multigroup_mode
    False
    >>>
    >>> # Configure time census with tally bins
    >>> s.set_time_census([1e-9, 1e-6], tally_frequency=10)
    >>> s.N_census
    3
    >>>
    >>> # Switch to eigenvalue mode
    >>> s.set_eigenmode(N_inactive=50, N_active=500, k_init=1.0)
    >>> s.eigenvalue_mode
    True
    """
    # Basic
    N_particle: int = 0
    N_batch: int = 1
    rng_seed: int = 1

    # Simulation mode
    multigroup_mode: bool = False
    eigenvalue_mode: bool = False

    # k-eigenvalue
    N_inactive: int = 0
    N_active: int = 0
    k_init: float = 1.0
    use_gyration_radius: bool = False
    gyration_radius_type: int = GYRATION_RADIUS_ALL

    # Particle source
    use_source_file: bool = False
    source_file_name: str = ""

    # Misc.
    time_boundary: float = np.inf
    output_name: str = "output"
    use_progress_bar: bool = True
    save_input_deck: bool = True

    # Time census
    N_census: int = 1
    census_time: NDArray[np.float64] = field(default_factory=lambda: np.array([np.inf]))
    use_census_based_tally: bool = False
    census_tally_frequency: int = 0

    # Particle bank-related
    save_particle: bool = False
    active_bank_buffer: int = 100
    census_bank_buffer_ratio: float = 1.0
    source_bank_buffer_ratio: float = 1.0
    future_bank_buffer_ratio: float = 0.5

    # Portability
    target_gpu: bool = False

    def __post_init__(self):
        """
        Finalize initialization, infer modes, and register the settings.

        Notes
        -----
        - Recasts selected fields to ``int`` for safety.
        - Sets ``multigroup_mode`` by inspecting the first material in
          ``mcdc.objects.materials``.
        - Registers this instance to ``mcdc.objects.settings``.
        """
        super().__init__("settings")

        # Recasting
        self.N_particle = int(self.N_particle)
        self.active_bank_buffer = int(self.active_bank_buffer)

        # Set multgroup mode
        self.multigroup_mode = isinstance(objects.materials[0], MaterialMG)
        
        # Register the settings
        objects.settings = self

    def set_time_census(self, time, tally_frequency=None):
        """
        Define time-census boundaries and optional census-based tallying.

        Parameters
        ----------
        time : array_like of float
            Strictly increasing time boundaries; must start > 0. The routine
            appends ``np.inf`` automatically as the terminal census.
        tally_frequency : int, optional
            Number of equal-width tally bins inside each census interval.
            If provided and > 0, enables census-based tallying and resets
            tallies' time meshes to the uniform grids implied.

        Notes
        -----
        - Validates that ``time`` is sorted and ``time[0] > 0.0``.
        - Sets ``N_census`` and ``census_time`` (with terminal ``inf``).

        Examples
        --------
        >>> s = Settings(N_particle=1000)
        >>> s.set_time_census([1e-9, 1e-6], tally_frequency=10)
        >>> s.N_census
        3
        >>> s.census_tally_frequency
        10
        """
        # Make sure that the time grid points are sorted
        if not is_sorted(time):
            print_error("Time census: Time grid points have to be sorted.")

        # Make sure that the starting point is larger than zero
        if time[0] <= 0.0:
            print_error("Time census: First census time should be larger than zero.")

        # Add the default, final census-at-infinity
        time = np.append(time, np.inf)

        # Set the time census parameters
        self.census_time = time
        self.N_census = len(self.census_time)

        # Set the census-based tallying
        if tally_frequency is not None and tally_frequency > 0:
            # Reset all tallies' time grids:
            self.use_census_based_tally = True
            self.census_tally_frequency = tally_frequency

    def set_eigenmode(
        self,
        N_inactive=0,
        N_active=0,
        k_init=1.0,
        gyration_radius=None,
        save_particle=False,
    ):
        """
        Enable and configure k-eigenvalue mode.

        Parameters
        ----------
        N_inactive : int, default 0
            Number of inactive cycles (discarded from k statistics).
        N_active : int, default 0
            Number of active cycles (used for k statistics).
        k_init : float, default 1.0
            Initial multiplication factor for power iteration.
        gyration_radius : {"all","infinite-x","infinite-y","infinite-z","only-x","only-y","only-z"}, optional
            Choose a gyration-radius type and enable ``use_gyration_radius``.
        save_particle : bool, default False
            Save final particle banks to output.

        Notes
        -----
        - Sets ``eigenvalue_mode=True`` and ``N_cycle=N_inactive+N_active``.
        - Unknown ``gyration_radius`` strings trigger :func:`print_error`.

        Examples
        --------
        >>> s = Settings()
        >>> s.set_eigenmode(N_inactive=50, N_active=500, k_init=1.0)
        >>> s.eigenvalue_mode
        True
        >>> s.N_cycle
        550
        """

        # Update setting self
        self.N_inactive = N_inactive
        self.N_active = N_active
        self.N_cycle = self.N_inactive + self.N_active
        self.eigenvalue_mode = True
        self.k_init = k_init
        self.save_particle = save_particle

        # Gyration radius setup
        if gyration_radius is not None:
            self.use_gyration_radius = True
            if gyration_radius == "all":
                self.gyration_radius_type = GYRATION_RADIUS_ALL
            elif gyration_radius == "infinite-x":
                self.gyration_radius_type = GYRATION_RADIUS_INFINITE_X
            elif gyration_radius == "infinite-y":
                self.gyration_radius_type = GYRATION_RADIUS_INFINITE_Y
            elif gyration_radius == "infinite-z":
                self.gyration_radius_type = GYRATION_RADIUS_INFINITE_Z
            elif gyration_radius == "only-x":
                self.gyration_radius_type = GYRATION_RADIUS_ONLY_X
            elif gyration_radius == "only-y":
                self.gyration_radius_type = GYRATION_RADIUS_ONLY_Y
            elif gyration_radius == "only-z":
                self.gyration_radius_type = GYRATION_RADIUS_ONLY_Z
            else:
                print_error("Unknown gyration radius type")

    def set_source_file(self, source_file_name):
        """
        Use an HDF5 particle source file and set ``N_particle`` accordingly.

        Parameters
        ----------
        source_file_name : str
            Path to the HDF5 file containing the dataset ``particles_size``.

        Notes
        -----
        - Sets ``use_source_file=True`` and stores the file name.
        - Reads ``particles_size`` from the file and assigns it to ``N_particle``.

        Examples
        --------
        >>> s = Settings()
        >>> s.set_source_file("source_particles.h5")
        >>> s.use_source_file
        True
        >>> s.N_particle
        10000  # depends on file contents
        """
        self.use_source_file = True
        self.source_file_name = source_file_name

        # Set number of particles
        with h5py.File(source_file_name, "r") as f:
            self.N_particle = int(f["particles_size"][()])
