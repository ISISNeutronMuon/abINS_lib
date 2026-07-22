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
    assert abinslib.data._get_pooch_or_none(abinslib.data._setup_validation_data) is None


def test_missing_pooch_error(monkeypatch):
    """Check for correct error message if no pooch available"""
    monkeypatch.setattr(abinslib.data, "_EUPHONIC_TEST_DATA", None)
    monkeypatch.setattr(abinslib.data, "_VALIDATION_DATA", None)

    with pytest.raises(ImportError, match="pooch"):
        abinslib.data.get_data("NaH.phonon")

    with pytest.raises(ImportError, match="pooch"):
        abinslib.data.get_validation_data("some_file.json")


def test_get_validation_data_local(tmp_path, monkeypatch):
    """Check that get_validation_data prioritizes local files."""
    dev_results_dir = tmp_path / "dev" / "validation" / "results"
    dev_results_dir.mkdir(parents=True)
    fake_file = dev_results_dir / "ethanol_mantid_isotropic_fundamentals.json"
    fake_file.write_text("dummy")

    # Monkeypatch the module's __file__ so local_dev_dir resolves to tmp_path/src/abinslib/data.py
    src_abinslib_dir = tmp_path / "src" / "abinslib"
    src_abinslib_dir.mkdir(parents=True)
    fake_module_file = src_abinslib_dir / "data.py"

    monkeypatch.setattr(abinslib.data, "__file__", str(fake_module_file))

    path = abinslib.data.get_validation_data("ethanol_mantid_isotropic_fundamentals.json")
    assert path == fake_file
    assert path.read_text() == "dummy"

def test_get_validation_data_pooch(monkeypatch):
    """Check that get_validation_data falls back to pooch if local file missing."""
    pytest.importorskip("pooch")

    # Force local file to not exist
    monkeypatch.setattr(abinslib.data, "__file__", "/does/not/exist/src/abinslib/data.py")

    # Mock the pooch fetch method
    class MockPooch:
        def fetch(self, filename):
            return f"/mock/pooch/path/{filename}"

    monkeypatch.setattr(abinslib.data, "_VALIDATION_DATA", MockPooch())

    path = abinslib.data.get_validation_data("ethanol_mantid_isotropic_fundamentals.json")
    assert str(path) == "/mock/pooch/path/ethanol_mantid_isotropic_fundamentals.json"

