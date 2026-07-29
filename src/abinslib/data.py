"""Wrapper to fetch reference data for tutorials etc."""

from __future__ import annotations

from collections.abc import Iterable
import importlib.resources
from pathlib import Path

try:
    import pooch
except ImportError:
    pooch = None


def _get_registry(
    cache_name: str, registry_filename: str, base_url: str = ""
) -> pooch.Pooch | None:
    if pooch is None:
        return None

    pooch_registry = pooch.create(
        path=pooch.os_cache(cache_name),
        base_url=base_url,
        registry=None,
    )
    registries = importlib.resources.files("abinslib.registries")
    registry_path = registries.joinpath(registry_filename)
    with registry_path.open("r") as fd:
        pooch_registry.load_registry(fd)
    return pooch_registry


_EUPHONIC_TEST_DATA: pooch.Pooch | None = _get_registry(
    "abinslib",
    "registry.txt",
    base_url="",  # URLs are defined inline in registry.txt
)
_VALIDATION_DATA: pooch.Pooch | None = _get_registry(
    "abinslib-validation",
    "registry_validation.txt",
    base_url="https://github.com/ISISNeutronMuon/abINS_lib/releases/download/validation-data-v1/",
)


def _setup_validation_search_dirs() -> tuple[Path, ...]:
    """Determine fallback search directories relative to the source tree if available.

    When installed from a wheel/sdist or as a zipped egg, these dev/ paths will not
    exist, but pathlib handles non-existent paths gracefully during `.is_file()` checks.
    """
    pkg_path = importlib.resources.files("abinslib")

    # If pkg_path is a zipfile/MultiplexedPath, we cannot safely use parents[1].
    if hasattr(pkg_path, "parents"):
        return (
            pkg_path.parents[1] / "dev" / "validation" / "results",
            pkg_path.parents[1] / "dev" / "validation" / "data",
        )
    return ()


VALIDATION_SEARCH_DIRS = _setup_validation_search_dirs()


def get_data(filename: str) -> Path:
    """Get external reference data by filename."""
    if _EUPHONIC_TEST_DATA is None:  # i.e. pooch dependency not available
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

    Args:
        filename:
            Target data filename. If downloaded from archive, hash will be
            checked against registry file.

        search_dirs:
            If provided, check these directories for local file with filename
            and skip hash check. If file is not found, fallback to archive
            download using pooch.

    """
    for search_dir in search_dirs:
        local_path = Path(search_dir, filename).resolve()
        if local_path.is_file():
            return local_path

    if _VALIDATION_DATA is None:  # i.e. pooch dependency not available
        msg = (
            "Could not construct validation data collection. Ensure 'pooch' was"
            " installed, e.g. with 'pip install abinslib[tutorials]'."
        )
        raise ImportError(msg)

    return Path(_VALIDATION_DATA.fetch(filename))
