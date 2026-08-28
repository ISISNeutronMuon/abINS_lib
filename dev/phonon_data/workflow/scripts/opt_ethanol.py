"""Compute optimised ethanol structure with PET-MAD."""

from pathlib import Path

from ase import Atoms
import ase.build
from ase.optimize import FIRE2, LBFGS
from snakemake.script import snakemake
from upet.calculator import UPETCalculator


def opt_ethanol(calc: UPETCalculator, traj: Path) -> Atoms:
    """Get initial structure from ASE and optimize with Calculator.

    This should be a "PET-MAD" model from https://pypi.org/project/upet/.
    These are general-purpose "foundation models" and won't give exact
    agreement with experiment (or even DFT)... but should be broadly physical
    and good enough for testing phonon methods.
    """
    # Make periodic cell for consistent phonon approach later
    atoms = ase.build.molecule("CH3CH2OH", vacuum=6.0, pbc=True)
    atoms.calc = calc

    opt = FIRE2(atoms, trajectory=traj)
    opt.run(steps=300, fmax=1e-3)

    opt = LBFGS(atoms, trajectory=traj)
    opt.run(steps=1000, fmax=1e-5)

    return atoms


def main():  # noqa: D103
    calculator = UPETCalculator(**snakemake.params.calc_params)

    atoms = opt_ethanol(calculator, traj=Path(snakemake.output[0]))
    atoms.write(snakemake.output[1])


if __name__ == "__main__":
    main()
