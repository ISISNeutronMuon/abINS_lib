"""Compare abinslib's convolution with Mantid's, on the same inputs (step 2).

Run in the abinslib environment after step 1 has written DATA_DIR::

    uv run python dev/validation/scripts/convolution_bench_abinslib.py [DATA_DIR]

Times the array-level routine on every atom of each case and prints it beside
the Mantid timings from step 1. Also checks that the new orders agree with
Mantid's (abinslib rows carry one bin width per order), and reports the
per-atom overhead of the public ``convolve_orders`` call separately.

The gate is checked at the largest mesh with 9 atoms: the array routine must be
no slower and use no more peak memory than Mantid's routine.
"""

import argparse
import json
from pathlib import Path

from benchmark_common import (
    default_data_dir,
    inputs_path,
    manifest_path,
    mantid_results_path,
    peak_bytes,
    time_min,
)
from euphonic import Quantity
from euphonic.spectra import Spectrum1D
import numpy as np

from abinslib.convolution import _convolve_orders_arrays, convolve_orders

OVERHEAD_ATOMS = 9  # atom count for the public-call overhead column
GATE_ATOMS = 9  # atom count for the regression gate


def _abinslib_all_atoms(
    fundamentals: np.ndarray, bin_width: float, n_new: int
) -> np.ndarray:
    """Autoconvolve every atom with the array routine, into one output array."""
    atoms, bins = fundamentals.shape
    out = np.empty((atoms, n_new, bins))
    for atom in range(atoms):
        values = fundamentals[atom]
        out[atom] = _convolve_orders_arrays(
            values, values, n_new, shift=0, spacing=bin_width
        )
    return out


def _public_all_atoms(spectra: list[Spectrum1D]) -> None:
    """Public convolve_orders for each atom, with the lower-edge reading stated."""
    for spectrum in spectra:
        convolve_orders(spectrum, spectrum, max_order=10, bin_position="lower_edge")


def _agreement(
    abinslib_rows: np.ndarray, mantid_rows: np.ndarray, bin_width: float
) -> float:
    """Max difference, relative to the peak, after removing the bin widths."""
    n_new = mantid_rows.shape[1]
    factors = bin_width ** np.arange(1, n_new + 1)
    scaled = abinslib_rows / factors[None, :, None]
    return float(np.abs(scaled - mantid_rows).max() / np.abs(mantid_rows).max())


def main() -> None:
    """Run the benchmark step and print a summary table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", nargs="?", type=Path, default=default_data_dir())
    args = parser.parse_args()
    data_dir = args.data_dir

    manifest = json.loads(manifest_path(data_dir).read_text())
    bin_width = float(manifest["bin_width"])
    max_order = int(manifest["max_order"])
    n_new = max_order - 1
    repeats = int(manifest["repeats"])
    mantid_cases = {(case["bins"], case["atoms"]): case for case in manifest["cases"]}

    header = (
        f"{'bins':>6} {'atoms':>5} | {'abinslib ms':>11} {'mantid ms':>10} "
        f"{'ratio':>6} | {'abinslib MiB':>12} {'mantid MiB':>10} | "
        f"{'max rel diff':>12} | {'public overhead us/atom':>23}"
    )
    print(header)
    print("-" * len(header))

    largest_bins = max(bins for bins, _ in mantid_cases)
    gate = None
    for (bins, atoms), mantid_case in sorted(mantid_cases.items()):
        with np.load(inputs_path(data_dir, bins, atoms)) as data:
            fundamentals = data["fundamentals"]
        with np.load(mantid_results_path(data_dir, bins, atoms)) as data:
            mantid_rows = data["new_orders"]

        def array_call(fundamentals=fundamentals):
            return _abinslib_all_atoms(fundamentals, bin_width, n_new)

        ours_seconds = time_min(array_call, repeats)
        ours_peak = peak_bytes(array_call)
        ours_rows = array_call()
        diff = _agreement(ours_rows, mantid_rows, bin_width)

        mantid_seconds = mantid_case["mantid_min_seconds"]
        mantid_peak = mantid_case["mantid_peak_bytes"]

        overhead_text = "-"
        if atoms == OVERHEAD_ATOMS:
            edges = np.arange(bins + 1) * bin_width
            spectra = [
                Spectrum1D(
                    Quantity(edges, "meV"),
                    Quantity(values, "barn / meV"),
                    metadata={"atom_index": index, "quantum_order": 1},
                )
                for index, values in enumerate(fundamentals)
            ]
            public_seconds = time_min(lambda: _public_all_atoms(spectra), repeats)
            overhead = (public_seconds - ours_seconds) / atoms
            overhead_text = f"{overhead * 1e6:23.0f}"

        print(
            f"{bins:6d} {atoms:5d} | {ours_seconds * 1e3:11.1f} "
            f"{mantid_seconds * 1e3:10.1f} "
            f"{ours_seconds / mantid_seconds:6.2f} | "
            f"{ours_peak / 2**20:12.1f} {mantid_peak / 2**20:10.1f} | "
            f"{diff:12.2e} | {overhead_text:>23}"
        )

        if bins == largest_bins and atoms == GATE_ATOMS:
            gate = (ours_seconds, mantid_seconds, ours_peak, mantid_peak)

    if gate is None:
        print("Gate not checked: largest mesh with 9 atoms is missing.")
        return
    ours_seconds, mantid_seconds, ours_peak, mantid_peak = gate
    passed = ours_seconds <= mantid_seconds and ours_peak <= mantid_peak
    print(
        "Gate (largest mesh, 9 atoms): "
        f"time {'ok' if ours_seconds <= mantid_seconds else 'FAILED'} "
        f"({ours_seconds * 1e3:.1f} vs {mantid_seconds * 1e3:.1f} ms), "
        f"memory {'ok' if ours_peak <= mantid_peak else 'FAILED'} "
        f"({ours_peak / 2**20:.1f} vs {mantid_peak / 2**20:.1f} MiB) -> "
        f"{'PASS' if passed else 'FAIL'}"
    )
    print(f"Times are minima over {repeats} repeats after one warm-up call.")


if __name__ == "__main__":
    main()
