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


def test_pooch_import_handler(monkeypatch):
    """Check _get_pooch_or_none returns None if no pooch available"""
    monkeypatch.setitem(sys.modules, "pooch", None)

    assert abinslib.data._get_pooch_or_none(abinslib.data._setup_ref_data) is None


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

    # Mock the pooch fetch method
    class MockPooch:
        def fetch(self, filename):
            return f"/mock/pooch/path/{filename}"

    monkeypatch.setattr(abinslib.data, "_VALIDATION_DATA", MockPooch())

    path = abinslib.data.get_validation_data(
        "ethanol_mantid_isotropic_fundamentals.json", search_dirs=[]
    )
    assert str(path) == "/mock/pooch/path/ethanol_mantid_isotropic_fundamentals.json"


def test_validation_search_dirs_zipped_fallback(monkeypatch):
    """Simulate importlib.resources.files return without .parents

    This is expected when working from a zipped package build, but testing is
    usually run from an editable install.
    """
    import importlib.resources
    import io

    class DummyMultiplexedPath:
        def joinpath(self, *args):
            class DummyFile:
                def open(self, mode):
                    return io.StringIO()

            return DummyFile()

    monkeypatch.setattr(importlib.resources, "files", lambda _: DummyMultiplexedPath())

    search_dirs = abinslib.data._setup_validation_search_dirs()
    assert search_dirs == ()
