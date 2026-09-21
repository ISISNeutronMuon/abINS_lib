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
        modes,
        b,
        a,
        q2,
        bins,
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
    ("temperature_k", "tosca_modes"),
    [
        (100, "ethanol"),
    ],
    indirect=["tosca_modes"],
)
def test_calculate_almost_isotropic_incoherent_combination_spectra(
    temperature_k, tosca_modes, ndarrays_regression
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
        modes,
        b,
        a,
        q2,
        bins,
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
    ("temperature_k", "tosca_modes"),
    [(100, "GaSb")],
    indirect=["tosca_modes"],
)
def test_q_scaling_almost_isotropic_incoherent_combination_spectra(
    temperature_k, tosca_modes, ndarrays_regression
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
        modes,
        b,
        a,
        q2,
        bins,
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
    ("temperature_k", "tosca_modes"),
    [(100, "GaSb")],
    indirect=["tosca_modes"],
)
def test_mantid_like_combination_spectra(
    temperature_k, tosca_modes, ndarrays_regression
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
        modes,
        b,
        a,
        q2,
        bins,
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
    ("temperature_k", "tosca_modes"),
    [(100, "GaSb")],
    indirect=["tosca_modes"],
)
def test_mantid_like_combination_spectra_collection_structure(
    temperature_k, tosca_modes
):
    """Test that mantid_like_combination_spectra returns correct collection."""
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
        modes=modes,
        mode_displacements=b,
        atomic_displacements=a,
        nominal_q2=q2,
        bins=bins,
    )

    # Get the number of atoms
    n_atoms = modes.crystal.n_atoms

    # Verify collection length equals atom count
    assert spectra.y_data.shape[0] == n_atoms, (
        f"Expected {n_atoms} spectra, got {spectra.y_data.shape[0]}"
    )

    # Verify 'qpt' is absent from metadata line_data
    assert "line_data" in spectra.metadata, "Expected 'line_data' in metadata"
    for line_item in spectra.metadata["line_data"]:
        assert "qpt" not in line_item, (
            f"'qpt' should be absent from line_data, but found: {line_item}"
        )

    # Verify atom attributes are preserved
    for i, line_item in enumerate(spectra.metadata["line_data"]):
        assert "atom_index" in line_item, f"Missing 'atom_index' in line_data[{i}]"
        assert "atom_symbol" in line_item, f"Missing 'atom_symbol' in line_data[{i}]"
        assert "mass" in line_item, f"Missing 'mass' in line_data[{i}]"


@pytest.mark.parametrize(
    ("temperature_k", "tosca_modes"),
    [(100, "GaSb")],
    indirect=["tosca_modes"],
)
def test_mantid_like_combination_spectra_invariance(
    temperature_k, tosca_modes, original_datadir
):
    """Test that the summed spectrum is invariant under flattening.

    This verifies that the physics is unchanged: the total spectrum obtained by
    summing all spectra across atoms should match the sum across all individual
    q-point and atom contributions from the old fixture.
    """
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

    # Get the new result with flattened collection
    new_spectra = mantid_like_combination_spectra(
        modes=modes,
        mode_displacements=b,
        atomic_displacements=a,
        nominal_q2=q2,
        bins=bins,
    )
    new_y = new_spectra.y_data.magnitude

    # Load the old fixture from Git-committed regression data
    fixture_name = "test_mantid_like_combination_spectra_100_GaSb_.npz"
    old_fixture_path = original_datadir / fixture_name
    old_data = np.load(old_fixture_path)
    old_y = old_data["y_data"]

    # Verify sum-invariance: new_y.sum(axis=0) should match old_y.sum(axis=0)
    new_sum = new_y.sum(axis=0)
    old_sum = old_y.sum(axis=0)

    np.testing.assert_allclose(
        new_sum,
        old_sum,
        atol=1e-12,
        rtol=1e-12,
        err_msg=(
            "Summed spectrum is not invariant under flattening; "
            "physics may have changed"
        ),
    )


@pytest.mark.parametrize(
    ("temperature_k", "tosca_modes"),
    [(100, "GaSb")],
    indirect=["tosca_modes"],
)
def test_iter_mantid_like_combination_qpt_spectra_yields_correct_shape(
    temperature_k, tosca_modes
):
    """Test that iterator yields N_q Spectrum1DCollections with N_a rows each."""
    from euphonic.spectra import Spectrum1DCollection

    from abinslib.almost_isotropic_incoherent import (
        _iter_mantid_like_combination_qpt_spectra,
    )
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

    # Get the number of q-points and atoms
    n_q = len(modes.weights)
    n_atoms = modes.crystal.n_atoms

    # Collect all yielded spectrum collections
    spectra_list = list(
        _iter_mantid_like_combination_qpt_spectra(
            modes=modes,
            mode_displacements=b,
            atomic_displacements=a,
            nominal_q2=q2,
            bins=bins,
        )
    )

    # Verify we get N_q collections
    assert len(spectra_list) == n_q, (
        f"Expected {n_q} collections, got {len(spectra_list)}"
    )

    # Verify each collection is a Spectrum1DCollection with shape (N_a, N_bins)
    for i, col in enumerate(spectra_list):
        assert isinstance(col, Spectrum1DCollection), (
            f"Item {i} is not a Spectrum1DCollection: {type(col)}"
        )
        assert col.y_data.shape[0] == n_atoms, (
            f"Collection {i} has {col.y_data.shape[0]} rows, expected {n_atoms}"
        )
