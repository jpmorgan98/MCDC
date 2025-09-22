import numpy as np

from numpy import float64
from numpy.typing import NDArray

####

import mcdc.objects as objects

from mcdc.constant import MATERIAL, MATERIAL_MG
from mcdc.nuclide import Nuclide
from mcdc.objects import ObjectOverriding
from mcdc.prints import print_1d_array, print_error

# ======================================================================================
# Material classes
# ======================================================================================


class MaterialBase(ObjectOverriding):
    """
    Base class for materials.

    Parameters
    ----------
    type_ : int
        Material type code (e.g., ``MATERIAL``, ``MATERIAL_MG``).
    name : str
        Optional material name.

    Attributes
    ----------
    ID : int
        Internal identifier from :class:`ObjectOverriding`.
    type : int
        Material type code.
    name : str
        Material name (may be empty).
    fissionable : bool
        True if any constituent enables fission.

    See Also
    --------
    Material
        Continuous-energy material defined by a nuclide composition.
    MaterialMG
        Multigroup material with groupwise cross sections.
    Nuclide
        Source of continuous-energy reaction data aggregated into materials.
    ReactionBase, ReactionNeutronCapture, ReactionNeutronElasticScattering, ReactionNeutronFission
        Reaction classes used by :class:`Nuclide`.
    DataTable, DataPolynomial, DataMultiPDF, DataMaxwellian
        Data containers used by reactions.
    """
    def __init__(self, type_, name):
        super().__init__("material", type_)
        self.name = name
        self.fissionable = False

    def __repr__(self):
        text = "\n"
        text += f"{decode_type(self.type)}\n"
        text += f"  - ID: {self.ID}\n"
        if self.name != '':
            text += f"  - Name: {self.name}\n"
        text += f"  - Fissionable: {self.fissionable}\n"
        return text


def decode_type(type_):
    if type_ == MATERIAL:
        return "Material"
    elif type_ == MATERIAL_MG:
        return "Multigroup material"


# ======================================================================================
# Native
# ======================================================================================


class Material(MaterialBase):
    """
    Continuous-energy material composed of named nuclides with number densities.

    Parameters
    ----------
    name : str, optional
        Material name (for reporting).
    nuclide_composition : dict, optional
        Mapping ``{nuclide_name: number_density}`` with density in
        atoms/(barn·cm). Each ``nuclide_name`` is loaded via :class:`Nuclide`.

    Attributes
    ----------
    nuclide_composition : dict[Nuclide, float]
        Mapping of instantiated :class:`Nuclide` objects to number densities.
    nuclides : list[Nuclide]
        Ordered list of constituent nuclides (Numba-friendly mirror).
    nuclide_densities : numpy.ndarray
        Densities aligned with ``nuclides`` (Numba-friendly mirror).
    fissionable : bool
        True if any constituent nuclide is fissionable.

    Notes
    -----
    - If a nuclide with the given name does not yet exist in ``mcdc.objects.nuclides``,
      it is created by calling :class:`Nuclide`.
    - The default value ``{}`` for ``nuclide_composition`` is mutable; consider
      passing an explicit dict (e.g., ``nuclide_composition=None`` and normalize inside)
      to avoid accidental state sharing.

    See Also
    --------
    Nuclide
        Loads continuous-energy reaction data from the HDF5 XS library.
    ReactionBase, ReactionNeutronCapture, ReactionNeutronElasticScattering, ReactionNeutronFission
        Reactions aggregated within each :class:`Nuclide`.
    DataTable, DataPolynomial, DataMultiPDF, DataMaxwellian
        Data containers used by the reactions.
    """
    def __init__(
        self,
        name: str = "",
        nuclide_composition: dict = {},
    ):
        type_ = MATERIAL
        super().__init__(type_, name)

        # Dictionary connecting nuclides to respective densities
        self.nuclide_composition = {}
        
        # For Numba mode: primitive nuclide_composition
        self.non_numba += ['nuclide_composition']
        self.nuclides = []
        self.nuclide_densities = np.zeros(len(nuclide_composition))

        # Loop over the items in the composition
        for i, (key, value) in enumerate(nuclide_composition.items()):
            nuclide_name = key
            nuclide_density = value

            # Check if nuclide is already created
            found = False
            for nuclide in objects.nuclides:
                if nuclide.name == nuclide_name:
                    found = True
                    break

            # Create the nuclide to objects if needed
            if not found:
                nuclide = Nuclide(nuclide_name)

            # Register the nuclide composition
            self.nuclides.append(nuclide)
            self.nuclide_densities[i] = nuclide_density
            self.nuclide_composition[nuclide] = nuclide_density

            # Some flags
            if nuclide.fissionable:
                self.fissionable = True


    def __repr__(self):
        """
        Return a human-readable summary of the material and its nuclide composition.

        Returns
        -------
        str
            A formatted multi-line string with ID, name, fissionable flag,
            and constituent nuclides with densities.
        """
        text = super().__repr__()
        text += f"  - Nuclide composition [atoms/barn-cm]\n"
        for nuclide in self.nuclide_composition.keys():
            text += f"    - {nuclide.name:<5} | {self.nuclide_composition[nuclide]}\n"
        return text
    

# ======================================================================================
# Multigroup
# ======================================================================================


class MaterialMG(MaterialBase):
    """
    Multigroup (MG) material with groupwise cross sections and spectra.

    Parameters
    ----------
    name : str, optional
        Material name.
    capture : (G,) arraylike of float, optional
        Groupwise capture cross section :math:`\\Sigma_c(g)`.
    scatter : (G, G) arraylike of float, optional
        Scattering transfer matrix :math:`\\Sigma_s(g\\to g')` laid out as
        ``[g_out, g_in]`` (will be transposed internally to ``[g_in, g_out]``
        and row-normalized to form ``chi_s``). Also summed by incoming group
        to produce ``Sigma_s``.
    fission : (G,) arraylike of float, optional
        Groupwise fission cross section :math:`\\Sigma_f(g)`.
    nu_s : (G,) arraylike of float, optional
        Scattering multiplication factor per group.
    nu_p : (G,) arraylike of float, optional
        Prompt neutron yield per fission, by incident group.
    nu_d : (J, G) arraylike of float, optional
        Delayed neutron yields, indexed ``[delayed_group, g_in]`` (will be
        transposed internally to ``[g_in, delayed_group]``).
    chi_p : (G,) or (G, G) arraylike of float, optional
        Prompt fission spectrum. If 1D, it is broadcast to columns to form (G,G),
        then transposed to ``[g_in, g_out]`` and row-normalized.
    chi_d : (G, J) arraylike of float, optional
        Delayed fission spectrum provided as ``[g_out, delayed_group]`` (will be
        transposed internally to ``[delayed_group, g_out]`` and row-normalized).
    speed : (G,) arraylike of float, optional
        Mean neutron speed per group.
    decay_rate : (J,) arraylike of float, optional
        Decay rate :math:`\\lambda_j` for each delayed group (1/s).

    Attributes
    ----------
    G : int
        Number of energy groups.
    J : int
        Number of delayed neutron groups (0 if none).
    mgxs_speed : (G,) ndarray
        Mean speeds.
    mgxs_decay_rate : (J,) ndarray
        Delayed group decay rates (``inf`` if none).
    mgxs_capture, mgxs_scatter, mgxs_fission, mgxs_total : (G,) ndarray
        Capture, (summed) scatter, fission, and total cross sections.
    mgxs_nu_s, mgxs_nu_p, mgxs_nu_d, mgxs_nu_d_total, mgxs_nu_f : ndarray
        Scattering multiplication; prompt/delayed yields; totals and combined yield.
    mgxs_chi_s, mgxs_chi_p, mgxs_chi_d : ndarray
        Scattering, prompt, and delayed spectra in internal orientation.
    fissionable : bool
        True if ``fission`` was provided (and thus production terms permitted).

    Notes
    -----
    - You must provide at least one of ``capture``, ``scatter``, or ``fission`` to
      define ``G``. If ``fission`` is provided and both ``nu_p`` and ``nu_d`` are
      missing, an error is reported via :func:`print_error`.
    - Internal transpositions normalize input layouts to
      ``chi_s : [g_in, g_out]``, ``chi_p : [g_in, g_out]``, and
      ``chi_d : [delayed_group, g_out]``.

    See Also
    --------
    Material
        Continuous-energy counterpart defined by nuclide composition.
    Nuclide
        Loads CE reaction data used to form materials.
    ReactionBase, ReactionNeutronCapture, ReactionNeutronElasticScattering, ReactionNeutronFission
        Reaction classes underlying CE materials.
    DataTable, DataPolynomial, DataMultiPDF, DataMaxwellian
        Data container types used by the reaction models.
    """
    def __init__(
        self,
        name: str = "",
        capture: NDArray[float64] = None,
        scatter: NDArray[float64] = None,
        fission: NDArray[float64] = None,
        nu_s: NDArray[float64] = None,
        nu_p: NDArray[float64] = None,
        nu_d: NDArray[float64] = None,
        chi_p: NDArray[float64] = None,
        chi_d: NDArray[float64] = None,
        speed: NDArray[float64] = None,
        decay_rate: NDArray[float64] = None,
    ):
        type_ = MATERIAL_MG
        super().__init__(type_, name)

        # Energy group size
        if capture is not None:
            G = len(capture)
        elif scatter is not None:
            G = len(scatter)
        elif fission is not None:
            G = len(fission)
        else:
            print_error("Need to supply capture, scatter, or fission for MaterialMG")
        self.G = G

        # Delayed group size
        J = 0
        if nu_d is not None:
            J = len(nu_d)
        self.J = J

        # Allocate the attributes
        self.mgxs_speed = np.ones(G)
        self.mgxs_decay_rate = np.ones(J) * np.inf
        self.mgxs_capture = np.zeros(G)
        self.mgxs_scatter = np.zeros(G)
        self.mgxs_fission = np.zeros(G)
        self.mgxs_total = np.zeros(G)
        self.mgxs_nu_s = np.ones(G)
        self.mgxs_nu_p = np.zeros(G)
        self.mgxs_nu_d = np.zeros([G, J])
        self.mgxs_nu_d_total = np.zeros([G])
        self.mgxs_nu_f = np.zeros(G)
        self.mgxs_chi_s = np.zeros([G, G])
        self.mgxs_chi_p = np.zeros([G, G])
        self.mgxs_chi_d = np.zeros([J, G])

        # Speed (vector of size G)
        if speed is not None:
            self.mgxs_speed = speed

        # Decay constant (vector of size J)
        if decay_rate is not None:
            self.mgxs_decay_rate = decay_rate

        # Cross-sections (vector of size G)
        if capture is not None:
            self.mgxs_capture = capture
        if scatter is not None:
            self.mgxs_scatter = np.sum(scatter, 0)
        if fission is not None:
            self.mgxs_fission = fission
            self.fissionable = True
        self.mgxs_total = self.mgxs_capture + self.mgxs_scatter + self.mgxs_fission

        # Scattering multiplication (vector of size G)
        if nu_s is not None:
            self.mgxs_nu_s = nu_s

        # Check if nu_p or nu_d is not provided, give fission
        if fission is not None:
            if nu_p is None and nu_d is None:
                print_error("Need to supply nu_p or nu_d for fissionable MaterialMG")

        # Prompt fission production (vector of size G)
        if nu_p is not None:
            self.mgxs_nu_p = nu_p

        # Delayed fission production (matrix of size GxJ)
        if nu_d is not None:
            # Transpose: [dg, gin] -> [gin, dg]
            self.mgxs_nu_d = np.swapaxes(nu_d, 0, 1)[:, :]
        self.mgxs_nu_d_total = np.sum(self.mgxs_nu_d, axis=1)

        # Total fission production (vector of size G)
        self.mgxs_nu_f = np.zeros_like(self.mgxs_nu_p)
        self.mgxs_nu_f += self.mgxs_nu_p
        for j in range(J):
            self.mgxs_nu_f += self.mgxs_nu_d[:, j]

        # Scattering spectrum (matrix of size GxG)
        if scatter is not None:
            # Transpose: [gout, gin] -> [gin, gout]
            self.mgxs_chi_s = np.swapaxes(scatter, 0, 1)[:, :]
            for g in range(G):
                if self.mgxs_scatter[g] > 0.0:
                    self.mgxs_chi_s[g, :] /= self.mgxs_scatter[g]

        # Prompt fission spectrum (matrix of size GxG)
        if nu_p is not None:
            if G == 1:
                self.mgxs_chi_p[:, :] = np.array([[1.0]])
            elif chi_p is None:
                print_error("Need to supply chi_p if nu_p is provided and G > 1")
            else:
                # Convert 1D spectrum to 2D
                if chi_p.ndim == 1:
                    tmp = np.zeros((G, G))
                    for g in range(G):
                        tmp[:, g] = chi_p
                    chi_p = tmp
                # Transpose: [gout, gin] -> [gin, gout]
                self.mgxs_chi_p[:, :] = np.swapaxes(chi_p, 0, 1)[:, :]
                # Normalize
                for g in range(G):
                    if np.sum(self.mgxs_chi_p[g, :]) > 0.0:
                        self.mgxs_chi_p[g, :] /= np.sum(self.mgxs_chi_p[g, :])

        # Delayed fission spectrum (matrix of size JxG)
        if nu_d is not None:
            if G == 1:
                self.mgxs_chi_d = np.ones([J, G])
            else:
                if chi_d is None:
                    print_error("Need to supply chi_d if nu_d is provided and G > 1")
                # Transpose: [gout, dg] -> [dg, gout]
                self.mgxs_chi_d = np.swapaxes(chi_d, 0, 1)[:, :]
            # Normalize
            for dg in range(J):
                if np.sum(self.mgxs_chi_d[dg, :]) > 0.0:
                    self.mgxs_chi_d[dg, :] /= np.sum(self.mgxs_chi_d[dg, :])

    def __repr__(self):
        """
        Return a summary of multigroup sizes and cross-section/spectrum arrays.

        Returns
        -------
        str
            A formatted multi-line string with G, J, main MGXS vectors/matrices,
            production terms, mean speeds, and decay constants.
        """
        text = super().__repr__()
        text += f"  - Multigroup data\n"
        text += f"    - G: {self.G}\n"
        text += f"    - J: {self.J}\n"
        text += f"    - Sigma_c {print_1d_array(self.mgxs_capture)}\n"
        text += f"    - Sigma_s {print_1d_array(self.mgxs_scatter)}\n"
        text += f"    - Sigma_f {print_1d_array(self.mgxs_fission)}\n"
        text += f"    - nu_s {print_1d_array(self.mgxs_nu_s)}\n"
        text += f"    - nu_p {print_1d_array(self.mgxs_nu_p)}\n"
        text += f"    - nu_d {print_1d_array(self.mgxs_nu_d.flatten())}\n"
        text += f"    - chi_s {print_1d_array(self.mgxs_chi_s.flatten())}\n"
        text += f"    - chi_fp {print_1d_array(self.mgxs_chi_p.flatten())}\n"
        text += f"    - chi_fd {print_1d_array(self.mgxs_chi_d.flatten())}\n"
        text += f"    - speed {print_1d_array(self.mgxs_speed)}\n"
        text += f"    - lambda {print_1d_array(self.mgxs_decay_rate)}\n"
        return text
