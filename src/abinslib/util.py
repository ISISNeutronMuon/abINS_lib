"""Utility functions, not specific to one calculation type."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from euphonic import Quantity
import numpy as np


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
