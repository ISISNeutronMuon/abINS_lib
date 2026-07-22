"""Wrapper to fetch reference data for tutorials etc."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pooch import Pooch

def _setup_ref_data() -> Pooch:
    import pooch

    return pooch.create(
        path=pooch.os_cache("abinslib"),
        base_url=(
            "https://github.com/pace-neutrons/Euphonic/raw/"
            "master/tests_and_analysis/test/data/"
        ),
        registry={
            "NaH.phonon": (
                "ccb30647b5cc9a2f3ab470dda77bc6f3ccc19cb1b8adaf35f5c40ccdaabccde1"
            ),
        },
        urls={
            "NaH.phonon": (
                "https://github.com/pace-neutrons/Euphonic/raw/master/"
                "tests_and_analysis/test/data/castep_files/NaH/NaH.phonon"
            ),
        },
    )

def _setup_validation_data() -> Pooch:
    import pooch

    return pooch.create(
        path=pooch.os_cache("abinslib-validation"),
        base_url="https://github.com/isisneutronmuon/abINS_lib/releases/download/validation-data-v1/",
        registry={
            "ethanol_mantid_isotropic_fundamentals.json": (
                "8d8a1fbe71e1d3ad98db96b9491da3e2d3464aa23bd0cec5c5da375b2f9ef30d"
            ),
            "ethanol_mantid_almost_isotropic_fundamentals.json": (
                "935abbea3534b76245b827a7c040f671510118a1800f9251f07dcd03e5b218d7"
            ),
            "ethanol_mantid_second_order.json": (
                "99f7daa152fcb626c18cd56f4d2a225b02d72ea5ee386ccc8017234fcad9a5b3"
            ),
            "ethanol_qpoint_phonon_modes.json": (
                "c0c3f306e44acec8db746e8e64c1af9936015759f0f76fe10e2d94101ad7a7f9"
            ),
        },
    )



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
