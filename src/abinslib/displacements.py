"""Functions for calculation of atomic displacement tensors.

The phonon mode displacements are described by 3x3 matrices related to the
covariance matrices of atomic positions associated with each mode. In the
CLIMAX/Abins notation this is 'B' and may include a Bose occupation factor.

The sum over modes (include Brillouin Zone integration for periodic systems)
with appropriate Bose occupation gives a physical distribution known in various
forms as 'A', 'Atomic Displacement Parameters', 'U', ... and sometimes even
'B'.

In abinslib these are referred to respectively as 'mode displacements' and
'atomic displacements' where space permits, or 'B' and 'A' in compact notation,
downcased to 'b' and 'a' in python variable names.

The Quantity A is equal to the debye_waller attribute of the DebyeWaller
class from Euphonic and may be read/stored from that class as convenient.

"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, TypedDict

from typing_extensions import Self

from euphonic import Quantity
import numpy as np

from .bose import BoseOccupation, calculate_bose_factor
from .io import JSONMixin

if TYPE_CHECKING:
    from euphonic import QpointPhononModes


class DisplacementsDict(TypedDict):
    """JSON-friendly representation of a Displacements object.

    Array indexing conventions:

    ============== =======================================
    Array          Index Order
    ============== =======================================
    displacements  (qpt, mode, atom, direction, direction)
    bose_n         (qpt, mode)
    weights        (qpt,)
    ============== =======================================

    """

    displacements: list[list[list[list[list[float]]]]]
    displacements_unit: str
    weights: list[float]
    bose_n: list[list[float]]
    temperature: float
    temperature_unit: str


@dataclass(frozen=True)
class Displacements(JSONMixin):
    """Phonon mode displacement dataset.

    This represents atomic displacements as 3x3 tensors, often denoted U or B.
    The data is arranged by qpoint, phonon mode and atom.

    Note that while the "displacements" attribute represents the underlying
    data of a Displacements object, the data should be accessed by the "one",
    "n", "n_plus_one" and "two_n_plus_one" properties which provide
    displacements at corresponding Bose occupation values.
    """

    displacements: Quantity
    """Atomic displacement tensor with dimensions (qpts, modes, atoms, 3, 3)
    and units length^2, without Bose occupation factor.
    (i.e. with the same values as the "one" property on resulting object.)
    """

    weights: np.ndarray
    """Normalised q-point weights corresponding to axis 0 of mode_displacements."""

    bose_n: np.ndarray
    """Bose factors <n> corresponding to displacements at target temperature.

    Other occupations <1>, <n+1> and <2n+1> will be derived from these values.
    """

    temperature: Quantity
    """Temperature for Bose occupation."""

    def __post_init__(self):
        """Make the underlying numpy array read-only so caching is safe."""
        self.displacements.setflags(write=False)

    @classmethod
    def from_modes(
        cls,
        modes: QpointPhononModes,
        temperature: Quantity,
        frequency_min: Quantity = Quantity(10, "cm_1"),
    ) -> Self:
        """Instantiate new Displacements from a QpointPhononModes.

        Bose occupation factors are calculated as <n> and stored along with
        unscaled displacements; the appropriate Bose scaling is applied later
        depending on the method used for retrieval.

        Args:
            modes: phonon mode data
            temperature: determines Bose factors and displacement magnitude
            frequency_min: frequency threshold below which mode displacements
                are set to zero. This is intended to eliminate translational
                modes at the Gamma point; for precise calculations on a fine
                q-point mesh it may be appropriate to reduce this threshold.

        """
        bose_factor = calculate_bose_factor(
            modes.frequencies,
            temperature,
            occupation=BoseOccupation.N,
        )

        return cls(
            displacements=_calculate_mode_displacements(
                modes,
                temperature,
                frequency_min,
            ),
            weights=modes.weights,
            bose_n=bose_factor,
            temperature=temperature,
        )

    def to_dict(self) -> DisplacementsDict:
        """Convert Displacements object to a JSON-friendly dictionary."""
        return {
            "displacements": self.displacements.magnitude.tolist(),
            "displacements_unit": str(self.displacements.units),
            "weights": self.weights.tolist(),
            "bose_n": self.bose_n.tolist(),
            "temperature": float(self.temperature.magnitude),
            "temperature_unit": str(self.temperature.units),
        }

    @classmethod
    def from_dict(cls, data_dict: DisplacementsDict) -> Self:
        """Create a Displacements object from a JSON-friendly dictionary."""
        displacements = Quantity(
            np.array(data_dict["displacements"], dtype=float),
            data_dict["displacements_unit"],
        )
        weights = np.array(data_dict["weights"], dtype=float)
        bose_n = np.array(data_dict["bose_n"], dtype=float)
        temperature = Quantity(data_dict["temperature"], data_dict["temperature_unit"])
        return cls(
            displacements=displacements,
            weights=weights,
            bose_n=bose_n,
            temperature=temperature,
        )

    @cached_property
    def one(self) -> Quantity:
        """Displacements without Bose occupation scaling."""
        return self.displacements

    @cached_property
    def n(self) -> Quantity:
        """Displacements with <n> Bose occupation scaling."""
        return np.einsum("ij,ij...->ij...", self.bose_n, self.displacements)

    @cached_property
    def n_plus_one(self) -> Quantity:
        """Displacements with <n+1> Bose occupation scaling."""
        return np.einsum("ij,ij...->ij...", self.bose_n + 1.0, self.displacements)

    @cached_property
    def two_n_plus_one(self) -> Quantity:
        """Displacements with <2n+1> Bose occupation scaling."""
        return np.einsum("ij,ij...->ij...", 2.0 * self.bose_n + 1.0, self.displacements)

    def to_atomic_displacements(self) -> Quantity:
        """Calculate atomic displacement tensor (A) for each atom.

        This is given in the same scaling convention as the "debye_waller"
        attribute of the DebyeWaller class in Euphonic, and may be used to
        initialize this class correctly. The value is half the "A" used in
        CLIMAX terminology, but for convenience the array is sometimes referred
        to as "a" or "A" in this codebase.

        In Euphonic coherent scattering intensity calculations, the
        Debye—Waller intensity factor appears as exp(-W_k) inside a square of
        sums.

        In incoherent intensity calculations the Debye—Waller factor typically
        appears as a pure factor exp(-2W_k).

        In both cases W_k is related to the displacement tensor of a given atom
        summed over all phonon modes.

        Returns:
          Array with length^2 dimensions,
          indices (atom_index, direction, direction)

          (i.e. a 3x3 quadratic displacement tensor per atom).

        """
        dw = np.einsum(
            "ijklm,i->klm",
            self.two_n_plus_one.magnitude,
            self.weights * 0.5,  # q-point symm weights, /2 scale convention for W
        )

        return Quantity(dw, self.displacements.units)


def _calculate_mode_displacements(
    modes: QpointPhononModes,
    temperature: Quantity,
    frequency_min: Quantity = Quantity(10, "cm_1"),
) -> Quantity:
    """Get the 3x3 displacement tensor (B) for each atom and phonon mode.

    Returns:
      Displacement array with length^2 dimensions and array indices: (qpt,
      mode, atom, direction, direction)

    """
    # Cast frequencies to atomic units -> displacement results in Bohr
    frequencies = modes.frequencies.to("hartree").magnitude

    # For very small frequencies scale by zero instead of inv freq.
    # (i.e. remove pure translation/rotation modes)
    inv_frequency = np.divide(
        1.0,
        frequencies,
        out=np.zeros_like(frequencies),
        where=(frequencies > frequency_min.to("hartree").magnitude),
    )

    evec_tensors = np.real(
        np.einsum("ijkl,ijkm->ijklm", modes.eigenvectors, np.conj(modes.eigenvectors))
    )

    mode_displacements = np.einsum(
        "ij,k,ijklm->ijklm",
        inv_frequency,
        1 / (2 * modes.crystal.atom_mass.to("atomic_unit_of_mass").magnitude),
        evec_tensors,
    )

    mode_displacements = Quantity(mode_displacements, "bohr**2").to(
        modes.crystal.cell_vectors_unit + "**2"
    )
    return mode_displacements
