"""Wrapper to fetch reference data for tutorials etc."""

from __future__ import annotations

from collections.abc import Callable
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
    reference_registry.load_registry(Path(__file__).with_name("registry.txt"))
    return reference_registry

def _setup_validation_data() -> Pooch:
    import pooch

    validation_registry = pooch.create(
        path=pooch.os_cache("abinslib-validation"),
        base_url="https://github.com/ISISNeutronMuon/abINS_lib/releases/download/validation-data-v1/",
        registry=None,
    )
    validation_registry.load_registry(Path(__file__).with_name("registry_validation.txt"))
    return validation_registry



def _get_pooch_or_none(setup_func: Callable[[], Pooch]) -> Pooch | None:
    try:
        import pooch  # noqa: F401
    except ImportError:
        return None
    return setup_func()


_EUPHONIC_TEST_DATA: Pooch | None = _get_pooch_or_none(_setup_ref_data)
_VALIDATION_DATA: Pooch | None = _get_pooch_or_none(_setup_validation_data)


def get_data(filename: str) -> Path:
    """Get external reference data by filename."""
    if _EUPHONIC_TEST_DATA is None:
        msg = (
            "Could not construct reference data collection. Ensure 'pooch' was"
            " installed, e.g. with 'pip install abinslib[tutorials]'."
        )
        raise ImportError(msg)

    return Path(_EUPHONIC_TEST_DATA.fetch(filename))


def get_validation_data(filename: str) -> Path:
    """Get validation reference data by filename.
    
    If the file exists locally in 'dev/validation/results' (relative to
    the project root), it is used as a priority over the remote archive.
    """
    # First, try to find it locally
    root_dir = Path(__file__).parent.parent.parent
    local_results_dir = root_dir / "dev" / "validation" / "results"
    local_data_dir = root_dir / "dev" / "validation" / "data"

    for search_dir in (local_results_dir, local_data_dir):
        local_path = search_dir / filename
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
