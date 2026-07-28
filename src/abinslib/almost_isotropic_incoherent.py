"""Semi-analytic powder averaging approximations in CLIMAX/AbINS lineage."""

from __future__ import annotations

from euphonic import QpointPhononModes, Quantity, ureg
from euphonic.spectra import Spectrum1DCollection
import numpy as np

from abinslib.util import iter_atom_info

from .displacements import Displacements
from .isotropic_incoherent import (
    _bin_mode_intensities,
    calculate_isotropic_dw_factor,
)


def calculate_almost_isotropic_incoherent_fundamentals(
    mode_displacements: Displacements,
    atomic_displacements: Quantity,
    nominal_q2: Quantity,
) -> np.ndarray:
    """Calculate fundamental mode intensities in almost-isotropic approximation.

    S =  exp(-(Q^2 tr(A + 2 tr(A:B)/tr(B))/5)) Q^2 tr(B) / 3

    - Fundamentals only
    - Atomic cross sections not applied
    - Ignore actual q-points and use nominal Q^2 instead

    Args:
        mode_displacements: phonon mode displacement dataset
        atomic_displacements: thermal average atomic displacements indexed
            (atom, direction, direction)
        nominal_q2:
            Scalar Q^2 values corresponding to modes; note that all q-points
            are used and this is typically related to the mode frequency by
            neutron instrument parameters.

    Returns:
        Dimensionless mode intensities with array indices (qpt, mode, atom)

    """
    q2_term = (
        np.einsum(
            "ij,ijkll->ijk",
            nominal_q2.to("bohr^-2").magnitude,
            mode_displacements.n_plus_one.to("bohr^2").magnitude,
        )
        / 3
    )

    a = atomic_displacements.to("bohr^2").magnitude * 2
    b = mode_displacements.n_plus_one.to("bohr^2").magnitude
    a_trace = np.einsum("...ii", a)
    b_trace = np.einsum("...ii", b)
    ba_trace = np.einsum("ijklm, kml->ijk", b, a)
    inv_b_trace = np.divide(
        1.0,
        b_trace,
        out=np.zeros_like(b_trace),
        where=(b_trace != 0.0),
    )

    exp_term = np.exp(
        -nominal_q2[:, :, None].to("bohr^-2").magnitude
        * (a_trace[None, None, :] + 2.0 * ba_trace * inv_b_trace)
        / 5.0
    )
    return q2_term * exp_term


def calculate_almost_isotropic_incoherent_combinations(
    mode_displacements: Displacements,
    atomic_displacements: Quantity,
    nominal_q2: Quantity,
    include_dw: bool = False,
) -> np.ndarray:
    """Calculate second-order mode intensities in almost-isotropic approximation.

    S(Q, ω_ν + ω_ν') =
        exp(-Q^2 tr(A/3)) Q^4 / 15C (tr(B_ν)tr(B_ν') + B_ν:B_ν' + B_ν':B_ν)

    for some atom, where C = 2 if ν=ν' else 1

    - Atomic cross sections not applied
    - Ignore actual q-points and use nominal Q^2 instead

    Note that this has cubic scaling with system size as n_modes ∝ n_atoms;
    while this reference implementation constructs the whole array,
    memory-efficient approaches need to reduce the data to binned spectra
    on-the-fly.

    It is also possible to reduce the calculation effort by calculating at
    constant Q and rescaling the intensity based on post-binning Q values;
    this is implemented in
    :func:`q_scaling_almost_isotropic_incoherent_combination_spectra`

    Args:
        mode_displacements: phonon mode displacement dataset
        atomic_displacements: thermal average atomic displacements indexed
            (atom, direction, direction)
        nominal_q2:
            Scalar Q^2 values corresponding to modes; note that all q-points
            are used and this is typically related to the mode frequency by
            neutron instrument parameters.
        include_dw:
            Include mode-by-mode Debye-Waller intensity scaling

    Returns:
        Dimensionless combination mode intensities with array indices
        (qpt1, mode1, qpt2, mode2, atom)

    """
    if len(nominal_q2.shape) != 4:
        msg = (
            "Expected 4-D nominal Q^2 for combination modes "
            "with indices (qpt1, mode1, qpt2, mode2)"
        )
        raise ValueError(msg)

    q4 = nominal_q2.to("bohr^-2").magnitude ** 2

    b = mode_displacements.n_plus_one.to("bohr^2").magnitude
    b_trace = np.einsum("...ii", b)
    tr_term = np.einsum("ijk,lmk->ijlmk", b_trace, b_trace)

    # Double contraction M_ij:M_ij should be commutative! Not clear why this is
    # traditionally written out as a sum over both orders, we can just *2?
    # Also as B are symmetric there is no difference between
    # B_ij:B_ij and B_ij:B_ji
    double_contraction = np.einsum("ijklm,opklm->ijopk", b, b)

    # Factor 2 if (q, ν) = (q', ν'), else 1
    n_q = b.shape[0]
    n_bands = b.shape[1]
    c = np.eye(n_q * n_bands, n_q * n_bands).reshape(
        n_q, n_bands, n_q, n_bands
    ) + np.ones((n_q, n_bands, n_q, n_bands))

    # No funny business here, just expand Q4 and C over atom index m
    q4_term = np.einsum(
        "ijkl,ijkl,ijklm->ijklm", q4, 1 / (15 * c), tr_term + 2 * double_contraction
    )

    if include_dw:
        dw_factor = calculate_isotropic_dw_factor(atomic_displacements, nominal_q2)
    else:
        dw_factor = 1.0

    return dw_factor * q4_term


def calculate_almost_isotropic_incoherent_spectra(
    modes: QpointPhononModes,
    mode_displacements: Displacements,
    atomic_displacements: Quantity,
    nominal_q2: Quantity,
    bins: Quantity,
) -> Spectrum1DCollection:
    """Calculate INS intensities in almost-isotropic incoherent approximation.

    Actual q-points of phonon modes will be disregarded; instead each mode
    intensity will be based on a separate array of nominal Q^2 values
    corresponding to modes. This is intended to approximate powder-averaging
    with kinematic constraints: for indirect geometry the energy-Q^2
    relationship can be determined using abinslib.utils.calculate_indirect_q2.

    Args:
        modes: phonon frequency and eigenvector dataset
        mode_displacements: phonon mode displacement dataset
            (This can be obtained using :func:`Displacements.from_modes(modes)`.)
        atomic_displacements: thermal average atomic displacements indexed
            (atom, direction, direction)
        nominal_q2:
            Scalar Q^2 values corresponding to modes; note that all q-points
            are used and this is typically related to the mode frequency by
            neutron instrument parameters.
        bins:
            Energy or frequency bins used as x_data in resulting spectra

    Returns:
        binned spectra of contribution from each nucleus

    """
    intensities = calculate_almost_isotropic_incoherent_fundamentals(
        mode_displacements=mode_displacements,
        atomic_displacements=atomic_displacements,
        nominal_q2=nominal_q2,
    )
    y_data = _bin_mode_intensities(
        modes=modes,
        intensities=intensities,
        bins=bins,
    )

    metadata = {
        "method": "almost-isotropic incoherent",
        "line_data": [
            item | {"quantum_order": 1} for item in iter_atom_info(modes.crystal)
        ],
    }
    spectra = Spectrum1DCollection(x_data=bins, y_data=y_data, metadata=metadata)
    return spectra


def calculate_almost_isotropic_incoherent_combination_spectra(
    modes: QpointPhononModes,
    mode_displacements: Displacements,
    atomic_displacements: Quantity,
    nominal_q2: Quantity,
    bins: Quantity,
) -> Spectrum1DCollection:
    """Calculate two-phonon intensities in almost-isotropic incoherent approximation.

    Actual q-points of phonon modes will be disregarded; instead each mode
    intensity will be based on a separate array of nominal Q^2 values
    corresponding to modes. This is intended to approximate powder-averaging
    with kinematic constraints: for indirect geometry the energy-Q^2
    relationship can be determined using abinslib.utils.calculate_indirect_q2.

    These should be determined for each two-phonon combination

    Args:
        modes: phonon frequency and eigenvector dataset
        mode_displacements: phonon mode displacement dataset
            (This can be obtained using :func:`Displacements.from_modes(modes)`.)
        atomic_displacements: thermal average atomic displacements indexed
            (atom, direction, direction)
        nominal_q2:
            Scalar Q^2 values for each combination of two fundamental modes, indexed by
            (q, band, q, band).  This is typically related to the combination mode
            frequency by neutron instrument parameters.
        bins:
            Energy or frequency bins used as x_data in resulting spectra

    Returns:
        binned spectra of contribution from each nucleus

    """
    intensities = calculate_almost_isotropic_incoherent_combinations(
        mode_displacements=mode_displacements,
        atomic_displacements=atomic_displacements,
        nominal_q2=nominal_q2,
        include_dw=True,
    )

    y_data = _bin_combination_modes(
        modes=modes,
        intensities=intensities,
        bins=bins,
    )

    metadata = {
        "method": "almost-isotropic incoherent",
        "line_data": [
            item | {"quantum_order": 2} for item in iter_atom_info(modes.crystal)
        ],
    }
    return Spectrum1DCollection(x_data=bins, y_data=y_data, metadata=metadata)


def q_scaling_almost_isotropic_incoherent_combination_spectra(
    modes: QpointPhononModes,
    mode_displacements: Displacements,
    atomic_displacements: Quantity,
    nominal_q2: Quantity,
    bins: Quantity,
) -> Spectrum1DCollection:
    """Calculate two-phonon intensities in almost-isotropic incoherent approximation.

    Actual q-points of phonon modes will be disregarded; instead each mode
    intensity will be based on a separate array of nominal Q^2 values
    corresponding to modes. This is intended to approximate powder-averaging
    with kinematic constraints.

    Here we also make the "optimisation" that intensities are initially
    calculated at Q=1 and then re-scaled after binning. (Not actually a big
    computational optimisation here as we still multiply a large Q2 array, but
    it imitates the Mantid implementation.)

    Args:
        modes: phonon frequency and eigenvector dataset
        mode_displacements: phonon mode displacement dataset
            (This can be obtained using :func:`Displacements.from_modes(modes)`.)
        atomic_displacements: thermal average atomic displacements indexed
            (atom, direction, direction)
        nominal_q2:
            Scalar Q^2 values corresponding to bin centres. For indirect geometry the
           energy-Q^2 relationship can be determined using
           abinslib.utils.calculate_indirect_q2.
        bins:
            Energy or frequency bins used as x_data in resulting spectra

    Returns:
        binned spectra of contribution from each nucleus

    """
    intensities = calculate_almost_isotropic_incoherent_combinations(
        mode_displacements=mode_displacements,
        atomic_displacements=atomic_displacements,
        nominal_q2=Quantity(
            np.ones((*modes.frequencies.shape, *modes.frequencies.shape)), "Å^-2"
        ),
        include_dw=False,
    )

    y_data = _bin_combination_modes(
        modes=modes,
        intensities=intensities,
        bins=bins,
    )

    metadata = {
        "method": "almost-isotropic incoherent approximation",
        "line_data": [
            item | {"quantum_order": 2} for item in iter_atom_info(modes.crystal)
        ],
    }
    spectra = Spectrum1DCollection(x_data=bins, y_data=y_data, metadata=metadata)

    q4_scale = nominal_q2**2 / Quantity(1, "Å^-4")
    dw = calculate_isotropic_dw_factor(
        atomic_displacements=atomic_displacements,
        q2=nominal_q2,
    )
    spectra.y_data = spectra.y_data * q4_scale * np.swapaxes(dw, -1, -2)[0]

    return spectra


def mantid_like_combination_spectra(
    modes: QpointPhononModes,
    mode_displacements: Displacements,
    atomic_displacements: Quantity,
    nominal_q2: Quantity,
    bins: Quantity,
) -> Spectrum1DCollection:
    """Calculate two-phonon intensities with approximations from Abins-Mantid.

    Currently the emphasis is on reproducibility, not efficiency.

    - DOS-like almost-isotropic incoherent approximation (i.e. semi-analytic
      powder-averaging equations with traces and contractions)
    - Calculate at nominal Q=1, rescale for Q4 relation and apply Debye-Waller
      _after_ binning
    - Treat each input q-point independently:
      - only consider combination modes at each q
      - weight each of these spectra with the weight of corresponding q
    - Order-2 scale factor is 1/60 for overtones and 1/30 for combinations

    - DW factor *is* still correctly averaged over q-point contributions

    Args:
        modes: phonon frequency and eigenvector dataset
        mode_displacements: phonon mode displacement dataset
            (This can be obtained using :func:`Displacements.from_modes(modes)`.)
        atomic_displacements: thermal average atomic displacements indexed
            (atom, direction, direction)
        nominal_q2:
            Scalar Q^2 values corresponding to bin centres. For indirect geometry the
           energy-Q^2 relationship can be determined using
           abinslib.utils.calculate_indirect_q2.
        bins:
            Energy or frequency bins used as x_data in resulting spectra

    Returns:
        binned spectra of contribution from each nucleus

    """
    spectra = Spectrum1DCollection(
        bins, np.empty((0, len(bins))) * ureg("barn") / bins.units
    )

    for q_index, weight in enumerate(modes.weights):
        qpt_modes = QpointPhononModes(
            crystal=modes.crystal,
            qpts=modes.qpts[np.newaxis, q_index],
            frequencies=modes.frequencies[np.newaxis, q_index],
            eigenvectors=modes.eigenvectors[np.newaxis, q_index],
            weights=np.array([1.0]),
        )
        qpt_displacements = Displacements(
            displacements=mode_displacements.displacements[np.newaxis, q_index],
            weights=np.array([1.0]),
            bose_n=mode_displacements.bose_n[np.newaxis, q_index],
            temperature=mode_displacements.temperature,
        )

        qpt_spectra = q_scaling_almost_isotropic_incoherent_combination_spectra(
            modes=qpt_modes,
            mode_displacements=qpt_displacements,
            atomic_displacements=atomic_displacements,
            nominal_q2=nominal_q2,
            bins=bins,
        )

        # Apply a couple of quirks from Mantid-Abins implementation:
        #
        # Exact origins/implications are being investigated, but these are
        # needed to reproduce established Mantid-Abins results (which agree
        # with expt well enough...)
        #
        # - Re-weight for current q-point (instead of product of weights)
        # - Apply 1/n! weighting on top of 1/15C mode weighting

        qpt_spectra.y_data = qpt_spectra.y_data * weight * 0.5
        qpt_spectra.metadata["qpt"] = str(modes.qpts[q_index])

        spectra = spectra + qpt_spectra

    spectra.group_by("atom_index")  # Combine q-point contributions

    return spectra


def _bin_combination_modes(
    modes: QpointPhononModes,
    intensities: np.ndarray,
    bins: Quantity,
) -> Quantity:
    """Bin intensities corresponding to QpointPhononModes to 1D spectra.

    This version is intended for the 2-phonon combination modes, so intensities
    has shape (q, band, q, band). Each contribution is weighted by the product
    of q-point weights and positioned at the sum of intensities

    Output array has shape (atom_indices, bin_indices)
    """
    bin_width = bins[1] - bins[0]

    if not np.isclose(modes.weights.sum(), 1):
        raise ValueError(
            "q-point weights sum to more than 1, this would lead to incorrect "
            "scaling between order-1 and order-2 spectra."
        )

    weighted_intensities = np.einsum(
        "i,k,ijklm->ijklm", modes.weights, modes.weights, intensities
    )

    frequencies = modes.frequencies.to(bins.units).magnitude
    combination_frequencies = (
        frequencies[:, :, None, None] + frequencies[None, None, :, :]
    )

    # reshape for single iteration of frequencies per atom
    combination_frequencies = combination_frequencies.reshape(-1)

    weighted_intensities = np.moveaxis(weighted_intensities, -1, 0)
    weighted_intensities = weighted_intensities.reshape(
        weighted_intensities.shape[0], -1
    )

    y_data = np.zeros([modes.crystal.n_atoms, len(bins) - 1])

    # Swap atom and freq axes for clean iteration
    for atom_index, atom_data in enumerate(weighted_intensities):
        y_q_atom, _ = np.histogram(
            combination_frequencies,
            bins=bins.magnitude,
            weights=atom_data,
            density=False,
        )
        y_data[atom_index] = y_q_atom

    # Apply correct spectral scaling / units
    return y_data / bin_width
