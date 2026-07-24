"""Unit tests for abinslib.util module"""

from euphonic import Quantity
from numpy import pi
from numpy.testing import assert_allclose
import pytest

from abinslib.util import calculate_indirect_q2


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
            (38.92 * pi / 180),
            Quantity(3.634, "meV"),
            Quantity([1.8674137643168969, 20.827728202858143], "1/Å"),
        ),
        (
            Quantity([11.496160804020064, 984.8698391959799], "meV").to("cm_1"),
            (38.92 * pi / 180),
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
