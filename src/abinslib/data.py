"""Wrapper to fetch reference data for tutorials etc."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import importlib.resources
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pooch import Pooch


def _setup_ref_data() -> Pooch:
    import pooch

    reference_registry = pooch.create(
        path=pooch.os_cache("abinslib"),
        base_url="",  # URLs are defined inline in registry.txt
        registry=None,
    )
    registries = importlib.resources.files("abinslib.registries")
    registry_path = registries.joinpath("registry.txt")
    with registry_path.open("r") as f:
        reference_registry.load_registry(f)
    return reference_registry


def _setup_validation_data() -> Pooch:
    import pooch

    validation_registry = pooch.create(
        path=pooch.os_cache("abinslib-validation"),
        base_url="https://github.com/ISISNeutronMuon/abINS_lib/releases/download/validation-data-v1/",
        registry=None,
    )
    registries = importlib.resources.files("abinslib.registries")
    registry_path = registries.joinpath("registry_validation.txt")
    with registry_path.open("r") as f:
        validation_registry.load_registry(f)
    return validation_registry


def _get_pooch_or_none(setup_func: Callable[[], Pooch]) -> Pooch | None:
    try:
        import pooch  # noqa: F401
    except ImportError:
        return None
    return setup_func()


_EUPHONIC_TEST_DATA: Pooch | None = _get_pooch_or_none(_setup_ref_data)
_VALIDATION_DATA: Pooch | None = _get_pooch_or_none(_setup_validation_data)

def _setup_validation_search_dirs() -> tuple[Path, ...]:
    """Determine fallback search directories relative to the source tree if available.
    
    When installed from a wheel/sdist or as a zipped egg, these dev/ paths will not
    exist, but pathlib handles non-existent paths gracefully during `.is_file()` checks.
    """
    _pkg_path = importlib.resources.files("abinslib")

    # If _pkg_path is a zipfile/MultiplexedPath, we cannot safely use parents[1].
    # We check if it has the standard pathlib `parents` attribute.
    if hasattr(_pkg_path, "parents"):
        return (
            _pkg_path.parents[1] / "dev" / "validation" / "results",
            _pkg_path.parents[1] / "dev" / "validation" / "data",
        )
    return ()

VALIDATION_SEARCH_DIRS = _setup_validation_search_dirs()


def get_data(filename: str) -> Path:
    """Get external reference data by filename."""
    if _EUPHONIC_TEST_DATA is None:
        msg = (
            "Could not construct reference data collection. Ensure 'pooch' was"
            " installed, e.g. with 'pip install abinslib[tutorials]'."
        )
        raise ImportError(msg)

    return Path(_EUPHONIC_TEST_DATA.fetch(filename))


def get_validation_data(
    filename: str,
    search_dirs: Iterable[Path | str] = VALIDATION_SEARCH_DIRS,
) -> Path:
    """Get validation reference data by filename.

    If `search_dirs` is provided, those directories are checked first.
    It defaults to checking the local development directories (e.g., 
    'dev/validation/results' relative to the package root).
    If the file is not found locally, it falls back to the Pooch remote archive.
    """
    for search_dir in search_dirs:
        local_path = (Path(search_dir) / filename).resolve()
        if local_path.is_file():
            return local_path

    # Fallback to pooch
    if _VALIDATION_DATA is None:
        msg = (
            "Could not construct validation data collection. Ensure 'pooch' was"
            " installed, e.g. with 'pip install abinslib[tutorials]'."
        )
        raise ImportError(msg)

    return Path(_VALIDATION_DATA.fetch(filename))
