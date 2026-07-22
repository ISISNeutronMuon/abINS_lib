from copy import deepcopy
from dataclasses import astuple

from euphonic import Quantity
import numpy as np
import pytest

from abinslib.almost_isotropic_incoherent import (
    calculate_almost_isotropic_incoherent_combination_spectra,
    calculate_almost_isotropic_incoherent_combinations,
    calculate_almost_isotropic_incoherent_fundamentals,
    calculate_almost_isotropic_incoherent_spectra,
    mantid_like_combination_spectra,
    q_scaling_almost_isotropic_incoherent_combination_spectra,
)


@pytest.mark.parametrize("tosca_modes", ["GaSb"], indirect=True)
def test_calculate_almost_isotropic_incoherent_fundamentals(
    tosca_modes, ndarrays_regression
):
    a, b = tosca_modes.ab(100)

    intensities = calculate_almost_isotropic_incoherent_fundamentals(
        mode_displacements=b, atomic_displacements=a, nominal_q2=tosca_modes.q2
    )
    ndarrays_regression.check({"intensities": intensities})


@pytest.mark.parametrize("tosca_modes", ["GaSb"], indirect=True)
def test_calculate_almost_isotropic_incoherent_combinations(
    tosca_modes, ndarrays_regression
):
    from abinslib.util import calculate_indirect_q2

    modes = tosca_modes.modes
    a, b = tosca_modes.ab(100)

    combination_frequencies = (
        modes.frequencies[:, :, None, None] + modes.frequencies[None, None, :, :]
    )

    q2 = calculate_indirect_q2(
        combination_frequencies,
        angle=(134.98885653282196 * np.pi / 180),
        final_energy=Quantity(32.0, "cm_1").to("hartree"),
    )

    intensities = calculate_almost_isotropic_incoherent_combinations(
        mode_displacements=b, atomic_displacements=a, nominal_q2=q2
    )
    assert intensities.shape == (
        *modes.frequencies.shape,
        *modes.frequencies.shape,
        modes.crystal.n_atoms,
    )
    ndarrays_regression.check({"intensities": intensities})


@pytest.mark.parametrize("tosca_modes", ["GaSb"], indirect=True)
def test_calculate_almost_isotropic_incoherent_combinations_bad_q(tosca_modes):
    modes = tosca_modes.modes
    a, b = tosca_modes.ab(100)

    bad_q2 = np.ones_like(modes.frequencies)

    with pytest.raises(ValueError, match="Expected 4-D"):
        calculate_almost_isotropic_incoherent_combinations(
            mode_displacements=b, atomic_displacements=a, nominal_q2=bad_q2
        )


@pytest.mark.parametrize(
    ("temperature_k", "tosca_modes"),
    [
        (10, "GaSb"),
        (100, "GaSb"),
        (10, "ethanol"),
        (100, "ethanol"),
    ],
    indirect=["tosca_modes"],
)
def test_calculate_almost_isotropic_incoherent_spectra(
    temperature_k, tosca_modes, ndarrays_regression
):
    """Test almost-isotropic fundamentals"""
    modes, q2 = astuple(tosca_modes)
    a, b = tosca_modes.ab(temperature_k)

    bins = Quantity(np.arange(0, 8000, 1), "cm_1")

    spectra = calculate_almost_isotropic_incoherent_spectra(
        modes, b, a, q2, bins,
    )

    ndarrays_regression.check(
        {
            "x_data": spectra.x_data.magnitude,
            "y_data": spectra.y_data.magnitude,
            "x_data_unit": spectra.x_data_unit,
            "y_data_unit": spectra.y_data_unit,
        }
    )


@pytest.mark.parametrize(
    ("temperature_k", "tosca_modes", "apply_cross_section"),
    [
        (100, "ethanol", False),
    ],
    indirect=["tosca_modes"],
)
def test_calculate_almost_isotropic_incoherent_combination_spectra(
    temperature_k, tosca_modes, apply_cross_section, ndarrays_regression
):
    """Test almost-isotropic fundamentals"""
    from abinslib.util import calculate_indirect_q2

    modes = tosca_modes.modes
    a, b = tosca_modes.ab(temperature_k)

    bins = Quantity(np.arange(0, 8000, 1), "cm_1")

    combination_frequencies = (
        modes.frequencies[:, :, None, None] + modes.frequencies[None, None, :, :]
    )

    q2 = calculate_indirect_q2(
        combination_frequencies,
        angle=(134.98885653282196 * np.pi / 180),
        final_energy=Quantity(32.0, "cm_1").to("hartree"),
    )

    spectra = calculate_almost_isotropic_incoherent_combination_spectra(
        modes, b, a, q2, bins, apply_cross_section=apply_cross_section
    )

    ndarrays_regression.check(
        {
            "x_data": spectra.x_data.magnitude,
            "y_data": spectra.y_data.magnitude,
            "x_data_unit": spectra.x_data_unit,
            "y_data_unit": spectra.y_data_unit,
        }
    )


@pytest.mark.parametrize("tosca_modes", ["ethanol"], indirect=["tosca_modes"])
def test_calculate_almost_isotropic_incoherent_combination_spectra_bad_weights(
    tosca_modes,
):
    """Test almost-isotropic fundamentals"""
    modes = deepcopy(tosca_modes.modes)
    modes.weights = np.ones_like(modes.weights)  # (i.e. sum > 1)
    a, b = tosca_modes.ab(100)

    bins = Quantity(np.arange(0, 8000, 1), "cm_1")

    q2 = Quantity(
        np.ones((*modes.frequencies.shape, *modes.frequencies.shape)),
        "bohr^-2",
    )

    with pytest.raises(ValueError, match="q-point weights sum to more than 1"):
        calculate_almost_isotropic_incoherent_combination_spectra(modes, b, a, q2, bins)


@pytest.mark.parametrize(
    ("temperature_k", "tosca_modes", "apply_cross_section"),
    [(100, "GaSb", False)],
    indirect=["tosca_modes"],
)
def test_q_scaling_almost_isotropic_incoherent_combination_spectra(
    temperature_k, tosca_modes, apply_cross_section, ndarrays_regression
):
    """Test almost-isotropic fundamentals"""
    from abinslib.util import calculate_indirect_q2

    modes = tosca_modes.modes
    a, b = tosca_modes.ab(temperature_k)

    bins = Quantity(np.arange(0, 8000, 1), "cm_1")

    bin_centres = (bins[1:] + bins[:-1]) * 0.5
    q2 = calculate_indirect_q2(
        bin_centres,
        angle=(134.98885653282196 * np.pi / 180),
        final_energy=Quantity(32.0, "cm_1").to("hartree"),
    )

    spectra = q_scaling_almost_isotropic_incoherent_combination_spectra(
        modes, b, a, q2, bins, apply_cross_section=apply_cross_section
    )

    ndarrays_regression.check(
        {
            "x_data": spectra.x_data.magnitude,
            "y_data": spectra.y_data.magnitude,
            "x_data_unit": spectra.x_data_unit,
            "y_data_unit": spectra.y_data_unit,
        }
    )


@pytest.mark.parametrize(
    ("temperature_k", "tosca_modes", "apply_cross_section"),
    [(100, "GaSb", False)],
    indirect=["tosca_modes"],
)
def test_mantid_like_combination_spectra(
    temperature_k, tosca_modes, apply_cross_section, ndarrays_regression
):
    """Test almost-isotropic fundamentals"""
    from abinslib.util import calculate_indirect_q2

    modes = tosca_modes.modes
    a, b = tosca_modes.ab(temperature_k)

    bins = Quantity(np.arange(0, 8000, 1), "cm_1")

    bin_centres = (bins[1:] + bins[:-1]) * 0.5
    q2 = calculate_indirect_q2(
        bin_centres,
        angle=(134.98885653282196 * np.pi / 180),
        final_energy=Quantity(32.0, "cm_1").to("hartree"),
    )

    spectra = mantid_like_combination_spectra(
        modes, b, a, q2, bins, apply_cross_section=apply_cross_section
    )

    ndarrays_regression.check(
        {
            "x_data": spectra.x_data.magnitude,
            "y_data": spectra.y_data.magnitude,
            "x_data_unit": spectra.x_data_unit,
            "y_data_unit": spectra.y_data_unit,
        }
    )
