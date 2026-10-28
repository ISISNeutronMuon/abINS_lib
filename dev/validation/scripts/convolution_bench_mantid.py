"""Time Mantid-Abins autoconvolution and store inputs and results (step 1).

Run this in the pixi environment, which provides Mantid and Euphonic 1.5::

    pixi run benchmark-convolution-mantid [DATA_DIR]

For each mesh size and atom count, seeded random order-1 spectra are built
and Mantid's ``add_autoconvolution_spectra`` is timed (single thread, no
resampling). Inputs, Mantid's new orders (2 to 10) and the timings are written
to DATA_DIR. The abinslib environment then runs step 2 on the same inputs::

    uv run python dev/validation/scripts/convolution_bench_abinslib.py [DATA_DIR]

Peak memory is measured in a separate traced call, so tracing does not affect
the timings.
"""

import argparse
import json
import os
from pathlib import Path
import tempfile

from benchmark_common import (
    ATOM_COUNTS,
    BIN_COUNTS,
    BIN_WIDTH,
    MAX_ORDER,
    REPEATS,
    SEED,
    default_data_dir,
    inputs_path,
    manifest_path,
    mantid_results_path,
    peak_bytes,
    time_min,
)
import numpy as np


def _configure_mantid_quietly() -> None:
    """Skip Mantid's startup instrument download and version check."""
    if "MANTIDCONFIGFOLDER" in os.environ:
        return
    config_dir = Path(tempfile.mkdtemp(prefix="mantid-config-"))
    (config_dir / "Mantid.user.properties").write_text(
        "UpdateInstrumentDefinitions.OnStartup = Off\n"
        "CheckMantidVersion.OnStartup = Off\n"
        "usagereports.enabled = 0\n"
    )
    os.environ["MANTIDCONFIGFOLDER"] = str(config_dir)


def _order_rows(result, atoms: int, bins: int) -> np.ndarray:
    """Mantid output as (atoms, orders 2..MAX_ORDER, bins), by metadata."""
    rows = np.full((atoms, MAX_ORDER - 1, bins), np.nan)
    for index, (line_meta, values) in enumerate(
        zip(result.iter_metadata(), result.y_data.magnitude, strict=True)
    ):
        order = int(line_meta["quantum_order"])
        if order >= 2:
            rows[int(line_meta["atom_index"]), order - 2] = values
    return rows


def main() -> None:
    """Run the benchmark step and print a summary table."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("data_dir", nargs="?", type=Path, default=default_data_dir())
    args = parser.parse_args()
    data_dir = args.data_dir
    data_dir.mkdir(parents=True, exist_ok=True)

    _configure_mantid_quietly()

    # Mantid must be imported (and configured) before the abins package
    import mantid.simpleapi  # noqa: F401  # isort:skip
    import abins.parameters  # isort:skip
    from abins.sdata import add_autoconvolution_spectra  # isort:skip
    from euphonic import Quantity  # isort:skip
    from euphonic.spectra import Spectrum1DCollection  # isort:skip

    abins.parameters.performance["threads"] = 1

    manifest = {
        "seed": SEED,
        "bin_width": BIN_WIDTH,
        "max_order": MAX_ORDER,
        "repeats": REPEATS,
        "cases": [],
    }
    for bins in BIN_COUNTS:
        for atoms in ATOM_COUNTS:
            rng = np.random.default_rng([SEED, bins, atoms])
            fundamentals = rng.random((atoms, bins))
            np.savez(
                inputs_path(data_dir, bins, atoms),
                fundamentals=fundamentals,
                bin_width=BIN_WIDTH,
            )

            edges = np.arange(bins + 1) * BIN_WIDTH
            spectra = Spectrum1DCollection(
                x_data=Quantity(edges, "meV"),
                y_data=Quantity(fundamentals, "barn / meV"),
                metadata={
                    "line_data": [
                        {"atom_index": atom, "quantum_order": 1}
                        for atom in range(atoms)
                    ]
                },
            )

            def call(spectra=spectra):
                return add_autoconvolution_spectra(spectra, max_order=MAX_ORDER)

            min_seconds = time_min(call, REPEATS)
            peak = peak_bytes(call)
            rows = _order_rows(call(), atoms, bins)
            np.savez(mantid_results_path(data_dir, bins, atoms), new_orders=rows)

            case = {
                "bins": bins,
                "atoms": atoms,
                "mantid_min_seconds": min_seconds,
                "mantid_peak_bytes": peak,
            }
            manifest["cases"].append(case)
            print(
                f"mantid bins={bins:6d} atoms={atoms:4d} "
                f"min={min_seconds * 1e3:10.1f} ms "
                f"peak={peak / 2**20:9.1f} MiB",
                flush=True,
            )

    manifest_path(data_dir).write_text(json.dumps(manifest, indent=2))
    print(f"Wrote inputs, Mantid results and timings to {data_dir}")


if __name__ == "__main__":
    main()
