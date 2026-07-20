from dataclasses import astuple
from pathlib import Path

from euphonic import Crystal, Quantity
import numpy as np
from numpy.testing import assert_allclose
import pytest

from abinslib.displacements import (
    Displacements,
)
import abinslib.isotropic_incoherent
from abinslib.isotropic_incoherent import (
    calculate_isotropic_dw_factor,
    calculate_isotropic_incoherent_spectra,
    q_scaling_isotropic_incoherent_spectra,
)
from abinslib.util import apply_weights

test_data = Path(__file__).parent / "data"


@pytest.fixture
def patch_cross_sections(monkeypatch):
    """Replace Euphonic cross-section lookup with Mantid values"""

    def _get_mantid_total_cross_sections(crystal: Crystal) -> Quantity:
        mantid_data = {
            "Ga": 6.83,
            "Sb": 3.9,
            "C": 5.551,
            "H": 82.02,
            "O": 4.232,
        }

        return Quantity([mantid_data[symbol] for symbol in crystal.atom_type], "barn")

    monkeypatch.setattr(
        abinslib.isotropic_incoherent,
        "_get_total_cross_sections",
        _get_mantid_total_cross_sections,
    )


@pytest.mark.parametrize(
    ("modes", "ref_npz"), [("GaSb", "GaSb_abins_isotropic_dw.npz")], indirect=True
)
def test_isotropic_dw(modes, ref_npz, ndarrays_regression):
    """Check isotropic DW on Q2 mesh agrees with Mantid-Abins"""
    q2 = Quantity(ref_npz["q2"], "angstrom^-2")

    a = Displacements.from_modes(
        modes, temperature=Quantity(100, "kelvin")
    ).to_atomic_displacements()

    # Fairly tight comparison with Mantid-Abins ref
    binned_dw_factor = calculate_isotropic_dw_factor(a, q2[None, :])
    assert_allclose(binned_dw_factor[0], ref_npz["iso_dw"].transpose(), rtol=1e-6)

    # Very tight comparison with regression data
    ndarrays_regression.check({"dw_factor": binned_dw_factor})


@pytest.mark.parametrize(
    ("temperature_k", "tosca_modes", "ref_npz"),
    [
        (10, "GaSb", "GaSb_abins_10k_isotropic_raw.npz"),
        (100, "GaSb", "GaSb_abins_100k_isotropic_raw.npz"),
        (10, "ethanol", "ethanol_abins_10k_isotropic_raw.npz"),
        (100, "ethanol", "ethanol_abins_100k_isotropic_raw.npz"),
    ],
    indirect=("tosca_modes", "ref_npz"),
)
def test_calculate_isotropic_incoherent_spectrum(
    temperature_k, tosca_modes, ref_npz, patch_cross_sections, ndarrays_regression
):
    """Test reference method for fully-isotropic calculation

    Note that there is some difference from Mantid-Abins reference because that
    implementation applies DW and Q2 scaling after initial energy binning
    """
    temperature = Quantity(temperature_k, "K")
    modes, q2 = astuple(tosca_modes)

    bins = Quantity(ref_npz["energy"], str(ref_npz["energy_unit"]))
    bin_width = bins[1] - bins[0]

    ref_intensity = ref_npz["intensity"]

    b = Displacements.from_modes(modes=modes, temperature=temperature)
    a = b.to_atomic_displacements()

    spectra = calculate_isotropic_incoherent_spectra(modes, b, a, q2, bins)
    spectra = apply_weights(spectra)  # default: Sears 1992 σ_tot
    spectrum = spectra.sum()

    # Loose check against Mantid-Abins: different quantisation scheme
    assert_allclose(
        spectrum.y_data.magnitude,
        ref_intensity[0] / bin_width.magnitude,
        rtol=1e-2,
    )

    # Exact check against regression data
    ndarrays_regression.check(
        {
            "y_data": spectrum.y_data.to("barn / meV").magnitude,
            "x_data": spectrum.x_data.to("meV").magnitude,
        }
    )


@pytest.mark.parametrize(
    ("temperature_k", "tosca_modes", "ref_npz"),
    [
        (10, "GaSb", "GaSb_abins_10k_isotropic_raw.npz"),
        (100, "GaSb", "GaSb_abins_100k_isotropic_raw.npz"),
        (10, "ethanol", "ethanol_abins_10k_isotropic_raw.npz"),
        (100, "ethanol", "ethanol_abins_100k_isotropic_raw.npz"),
    ],
    indirect=("tosca_modes", "ref_npz"),
)
def test_q_scaling_isotropic_incoherent_spectrum(
    temperature_k,
    tosca_modes,
    ref_npz,
    patch_cross_sections,
    ndarrays_regression,
):
    """Validate fully-isotropic calculation against Mantid-Abins data

    This implementation follows the scheme of rescaling bins for DW/Q2 terms

    The difference with Mantid-Abins reference is smaller than exact
    calculation, if still larger than expected
    """
    bins = Quantity(ref_npz["energy"], str(ref_npz["energy_unit"]))
    bin_width = bins[1] - bins[0]

    ref_intensity = ref_npz["intensity"]

    modes = tosca_modes.modes
    a, b = tosca_modes.ab(temperature_k)

    q2 = Quantity(np.load(test_data / "abins-q2-1_4-dump.npy"), "Å^-2")

    spectra = q_scaling_isotropic_incoherent_spectra(modes, b, a, q2, bins)
    spectra = apply_weights(spectra)  # default: Sears 1992 σ_tot
    spectrum = spectra.sum()

    # Fairly tight check against Mantid-Abins reference
    assert_allclose(
        spectrum.y_data.magnitude,
        ref_intensity[0] / bin_width.magnitude,
        rtol=1e-8,
    )

    # Very tight check against regression data
    ndarrays_regression.check({"y_data": spectrum.y_data.magnitude})
