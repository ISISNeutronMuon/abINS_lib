"""Unit tests for abinslib.util module"""

from copy import deepcopy

from euphonic import Quantity, ureg
from euphonic.spectra import Spectrum1DCollection
import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from abinslib.util import _AtomSequence, apply_weights, calculate_indirect_q2


def test_get_version(monkeypatch):
    """Test get_version returns installed version or 'DEVELOPMENT' fallback"""
    from importlib.metadata import PackageNotFoundError

    import abinslib.util

    # Simulate installed package case
    monkeypatch.setattr("abinslib.util.version", lambda pkg: "1.2.3")
    assert abinslib.util.get_version() == "1.2.3"

    # Simulate uninstalled package case
    def mock_version_not_found(pkg):
        raise PackageNotFoundError

    monkeypatch.setattr("abinslib.util.version", mock_version_not_found)
    assert abinslib.util.get_version() == "DEVELOPMENT"


@pytest.mark.parametrize(
    ("energy_transfer", "angle", "final_energy", "expected_q"),
    [
        (
            Quantity([11.496160804020064, 984.8698391959799], "meV"),
            (38.92 * np.pi / 180),
            Quantity(3.634, "meV"),
            Quantity([1.8674137643168969, 20.827728202858143], "1/Å"),
        ),
        (
            Quantity([11.496160804020064, 984.8698391959799], "meV").to("cm_1"),
            (38.92 * np.pi / 180),
            Quantity(3.634, "meV"),
            Quantity([1.8674137643168969, 20.827728202858143], "1/Å"),
        ),
    ],
)
def test_calculate_indirect_q2(
    energy_transfer, angle, expected_q, final_energy
) -> None:
    """Check indirect-geometry kinematic-constraint calculation

    Reference values and TOSCA parameters are from the Mantid 6.15 QECoverage
    interface

    (Note that the nominal 38.92 scattering angle is quite a large deviation
     from the 45 degrees currently used for intensity calculations.)
    """

    q2 = calculate_indirect_q2(
        energy_transfer=energy_transfer,
        angle=angle,
        final_energy=final_energy,
    )

    assert_allclose(q2.to("Å^-2").magnitude, (expected_q**2).to("Å^-2").magnitude)


@pytest.fixture
def h2d_spectra(rng) -> Spectrum1DCollection:
    x_data = Quantity(np.linspace(0, 5, 6), "meV")
    y_data = Quantity(rng.random((3, 5)), "1/meV")

    return Spectrum1DCollection(
        x_data=Quantity(np.linspace(0, 5, 6), "meV"),
        y_data=Quantity(rng.random((3, 5)), "1/meV"),
        metadata={
            "line_data": [
                {"atom_index": 1, "mass": "1.01"},
                {"atom_index": 2, "mass": "1.01"},
                {"atom_index": 3, "mass": "2.0"},
            ],
            "atom_symbol": "H",
        },
    )

def test_atom_sequence(h2d_spectra) -> None:
    atoms = _AtomSequence.from_spectra(h2d_spectra)

    assert_array_equal(atoms.atom_type, ["H", "H", "H"])
    assert_allclose(atoms.atom_mass.magnitude, [1.01, 1.01, 2.0])
    assert atoms.atom_mass.units == ureg("amu")


def test_apply_weights(h2d_spectra) -> None:
    weighted = apply_weights(h2d_spectra)

    assert weighted.y_data.units == ureg("barn / meV")
    assert_allclose(weighted.y_data[:2].magnitude, h2d_spectra.y_data[:2].magnitude * 82.02)
    assert_allclose(weighted.y_data[2].magnitude, h2d_spectra.y_data[2].magnitude * 7.64)


def test_bad_spectra(h2d_spectra) -> None:
    ref_metadata = deepcopy(h2d_spectra.metadata)

    del h2d_spectra.metadata["atom_symbol"]
    with pytest.raises(
        ValueError, match="Not all items in spectra have atom_symbol and mass metadata."):
        apply_weights(h2d_spectra)

    h2d_spectra.metadata = deepcopy(ref_metadata)
    h2d_spectra.metadata["line_data"][1] = {}
    with pytest.raises(
        ValueError, match="Not all items in spectra have atom_symbol and mass metadata."):
        apply_weights(h2d_spectra)
