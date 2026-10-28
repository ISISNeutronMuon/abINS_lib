"""Shared settings and file layout for the convolution benchmark.

Used by ``convolution_bench_mantid.py`` (pixi environment) and
``convolution_bench_abinslib.py`` (abinslib environment). Standard library and
NumPy only, so it imports in both environments.
"""

from collections.abc import Callable
from pathlib import Path
import tempfile
import time
import tracemalloc

SEED = 20260101
BIN_WIDTH = 0.5  # meV; energy zero sits on the lower edge of the first bin
BIN_COUNTS = (2000, 20000, 37280)
ATOM_COUNTS = (9, 100)
MAX_ORDER = 10  # Mantid-Abins default; gives 9 new orders from order 1
REPEATS = 5  # timed repeats; the minimum is reported


def default_data_dir() -> Path:
    """Directory for inputs, Mantid results and timings (outside the repo)."""
    return Path(tempfile.gettempdir()) / "abinslib-convolution-benchmark"


def manifest_path(data_dir: Path) -> Path:
    """JSON file with the benchmark settings and Mantid timings."""
    return data_dir / "manifest.json"


def inputs_path(data_dir: Path, bins: int, atoms: int) -> Path:
    """NumPy archive with the seeded fundamentals for one case."""
    return data_dir / f"inputs_{bins}_{atoms}.npz"


def mantid_results_path(data_dir: Path, bins: int, atoms: int) -> Path:
    """NumPy archive with Mantid's new orders for one case."""
    return data_dir / f"mantid_{bins}_{atoms}.npz"


def time_min(call: Callable[[], object], repeats: int = REPEATS) -> float:
    """Minimum wall time of ``call`` over repeats, after one warm-up call."""
    call()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        call()
        times.append(time.perf_counter() - start)
    return min(times)


def peak_bytes(call: Callable[[], object]) -> int:
    """Peak traced allocation during one call (separate from timing runs)."""
    tracemalloc.start()
    try:
        call()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak
