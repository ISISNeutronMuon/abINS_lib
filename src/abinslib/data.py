"""Wrapper to fetch reference data for tutorials etc."""

from __future__ import annotations

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
            )
        },
        urls={
            "NaH.phonon": (
                "https://github.com/pace-neutrons/Euphonic/raw/master/"
                "tests_and_analysis/test/data/castep_files/NaH/NaH.phonon"
            )
        },
    )

def _setup_validation_data() -> Pooch:
    import pooch

    return pooch.create(
        path=pooch.os_cache("abinslib-validation"),
        base_url="https://github.com/isisneutronmuon/abINS_lib/releases/download/validation-data-v1/",
        registry={
            "ethanol_mantid_isotropic_fundamentals.json": None,
            "ethanol_mantid_almost_isotropic_fundamentals.json": None,
            "ethanol_mantid_second_order.json": None,
        },
    )

def _ref_data_or_none() -> Pooch | None:
    try:
        import pooch  # noqa: F401
    except ImportError:
        return None

    return _setup_ref_data()

def _validation_data_or_none() -> Pooch | None:
    try:
        import pooch  # noqa: F401
    except ImportError:
        return None

    return _setup_validation_data()


_EUPHONIC_TEST_DATA: Pooch | None = _ref_data_or_none()
_VALIDATION_DATA: Pooch | None = _validation_data_or_none()


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
    local_dev_dir = root_dir / "dev" / "validation" / "results"
    local_path = local_dev_dir / filename
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
