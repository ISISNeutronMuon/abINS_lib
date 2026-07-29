import importlib
import importlib.resources
from io import StringIO
from pathlib import Path
import sys

import pytest

import abinslib.data


def test_get_data():
    """If pooch is available, check that reference data is loaded correctly"""
    pytest.importorskip("pooch")

    nah = abinslib.data.get_data("NaH.phonon")
    assert isinstance(nah, Path)
    assert nah.is_file()
    assert nah.name == "NaH.phonon"

    with nah.open() as fd:
        header = next(fd).strip()
        assert header == "BEGIN header"


def test_validation_registry_entries():
    """Verify that reference registries load expected files into Pooch."""
    pytest.importorskip("pooch")

    ref_data = abinslib.data._EUPHONIC_TEST_DATA
    val_data = abinslib.data._VALIDATION_DATA

    assert ref_data is not None
    assert "NaH.phonon" in ref_data.registry

    assert val_data is not None
    assert "ethanol_mantid_isotropic_fundamentals.json" in val_data.registry
    assert "ethanol_qpoint_phonon_modes.json" in val_data.registry


def test_pooch_import_handler(monkeypatch):
    """Check _get_registry returns None if no pooch available."""
    monkeypatch.setattr(abinslib.data, "pooch", None)

    assert abinslib.data._get_registry("abinslib", "registry.txt") is None


def test_pooch_module_import_failure(monkeypatch):
    """Check module load behavior when pooch import fails."""
    monkeypatch.setitem(sys.modules, "pooch", None)
    importlib.reload(abinslib.data)

    assert abinslib.data.pooch is None
    assert abinslib.data._EUPHONIC_TEST_DATA is None
    assert abinslib.data._VALIDATION_DATA is None

    monkeypatch.undo()
    importlib.reload(abinslib.data)


def test_missing_pooch_error(monkeypatch):
    """Check for correct error message if no pooch available"""
    monkeypatch.setattr(abinslib.data, "_EUPHONIC_TEST_DATA", None)
    monkeypatch.setattr(abinslib.data, "_VALIDATION_DATA", None)

    with pytest.raises(ImportError, match="pooch"):
        abinslib.data.get_data("NaH.phonon")

    with pytest.raises(ImportError, match="pooch"):
        abinslib.data.get_validation_data("some_file.json")


def test_get_validation_data_local(tmp_path):
    """Check that get_validation_data prioritizes local files when given search_dirs."""
    fake_file = tmp_path / "ethanol_mantid_isotropic_fundamentals.json"
    fake_file.write_text("dummy")

    path = abinslib.data.get_validation_data(
        "ethanol_mantid_isotropic_fundamentals.json", search_dirs=[tmp_path]
    )
    assert path == fake_file
    assert path.read_text() == "dummy"


def test_get_validation_data_pooch(monkeypatch):
    """Check that get_validation_data falls back to pooch if local file missing."""
    pytest.importorskip("pooch")

    # Mock fetch on the real Pooch object to verify registry lookup passes
    monkeypatch.setattr(
        abinslib.data._VALIDATION_DATA,
        "fetch",
        lambda filename: f"/mock/pooch/path/{filename}",
    )

    path = abinslib.data.get_validation_data(
        "ethanol_mantid_isotropic_fundamentals.json", search_dirs=[]
    )
    assert str(path) == "/mock/pooch/path/ethanol_mantid_isotropic_fundamentals.json"


def test_validation_search_dirs_zipped_fallback(monkeypatch):
    """Simulate importlib.resources.files return without .parents

    This is expected when working from a zipped package build, but testing is
    usually run from an editable install.
    """

    class DummyMultiplexedPath:
        def joinpath(self, *args):
            class DummyFile:
                def open(self, mode):
                    return StringIO()

            return DummyFile()

    monkeypatch.setattr(importlib.resources, "files", lambda _: DummyMultiplexedPath())

    search_dirs = abinslib.data._setup_validation_search_dirs()
    assert search_dirs == ()
