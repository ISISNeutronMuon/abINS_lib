"""Utility functions, not specific to one calculation type."""


from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Self

from euphonic import Quantity
from euphonic.isotopes import sears_1992
from euphonic.spectra import Spectrum1DCollection
import numpy as np

if TYPE_CHECKING:
    from euphonic.isotopes import IsotopeData


def is_none(a: object) -> bool:
    """Backport from python 3.14 operator.is_none."""
    return a is None


def get_version() -> str:
    """Get package version or return 'DEVELOPMENT' if not installed."""
    try:
        return version("abinslib")
    except PackageNotFoundError:
        return "DEVELOPMENT"


def calculate_indirect_q2(
    energy_transfer: Quantity, angle: float, final_energy: Quantity
) -> Quantity:
    """Calculate Q^2 value for given energy transfer in indirect geometry.

    By the cosine law Q^2 = k_f^2 + k_i^2 - 2 k_f k_i cos(theta)

    Args:
        energy_transfer: neutron energy change. (Positive values correspond to
            transfer to sample.)

        angle: scattering angle in radians

        final_energy: energy of detected neutrons (i.e. after monochromator)

    Returns:
        array of scalar Q^2 corresponding to input energy_transfer

    """
    # Get rid of ambiguous cm-1 units before manipulating energies
    energy_transfer = energy_transfer.to("meV", "spectroscopy")
    final_energy = final_energy.to("meV", "spectroscopy")

    # E = hbar^2 k^2 / 2m
    momentum2_to_energy = Quantity(0.5, "hbar^2 / neutron_mass").to("meV Å^2")

    k2_i = (energy_transfer + final_energy) / momentum2_to_energy
    k2_f = final_energy / momentum2_to_energy
    return k2_i + k2_f - 2 * np.sqrt(k2_i * k2_f) * np.cos(angle)


@dataclass
class _AtomSequence:
    atom_type: np.ndarray
    atom_mass: Quantity

    @classmethod
    def from_spectra(cls, spectra: Spectrum1DCollection) -> Self:
        """Build a quasi-structure object from metadata of spectra."""
        symbols = [item.get("atom_symbol") for item in spectra.iter_metadata()]
        masses = [item.get("mass") for item in spectra.iter_metadata()]

        if any(map(is_none, symbols)) or any(map(is_none, masses)):
            raise ValueError(
                "Not all items in spectra have atom_symbol and mass metadata."
            )

        atom_type = np.array(symbols)
        atom_mass = Quantity(np.array(masses, dtype=float), "amu")

        return cls(atom_type, atom_mass)


def apply_weights(
    spectra: Spectrum1DCollection,
    isotope_data: IsotopeData = sears_1992,
    key: str = "scattering_cross_section",
) -> Spectrum1DCollection:
    """Apply weights to Spectrum Collection data, based on atom symbol and mass.

    Initially this only supports Spectrum1DCollection, but support for
    Spectrum2DCollection will be added as needed.

    Args:
        spectra: unweighted data including 'atom_symbol' and 'mass' metadata
        isotope_data: neutron dataset with symbol/mass lookup capability
        key: key for get_array() lookups in isotope_data

    Returns:
        new set of weighted spectra

    """
    atoms = _AtomSequence.from_spectra(spectra)
    weights = isotope_data.get_array(atoms, key=key)

    y_data = spectra.y_data * weights[:, None]

    return Spectrum1DCollection(
        x_data=spectra.x_data, y_data=y_data, metadata=spectra.metadata
    )
