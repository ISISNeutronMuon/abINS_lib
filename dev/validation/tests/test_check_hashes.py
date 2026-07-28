"""Tests for validation script check_hashes."""

from io import StringIO
from pathlib import Path
import sys

import pytest

from dev.validation.scripts.check_hashes import (
    compare_hashes,
    get_parser,
    main,
    read_hashes,
)


def test_read_hashes(tmp_path: Path):
    hash_file = tmp_path / "hashes.txt"
    hash_file.write_text(
        "# Header comment\nfile1.json hash1\n\nfile2.json hash2 extra_info\n"
    )

    result = read_hashes(hash_file)
    assert result == {"file1.json": "hash1", "file2.json": "hash2"}


def test_read_hashes_stdin(monkeypatch):
    monkeypatch.setattr(
        sys,
        "stdin",
        StringIO("# Comment\nfile1.json hash1\nfile2.json hash2\n"),
    )
    result = read_hashes(Path("-"))
    assert result == {"file1.json": "hash1", "file2.json": "hash2"}


def test_compare_hashes_matching():
    new_h = {"file1.json": "hash1", "file2.json": "hash2"}
    ref_h = {"file1.json": "hash1", "file2.json": "hash2"}

    mismatches = compare_hashes(new_h, ref_h)
    assert mismatches == []


def test_compare_hashes_mismatch_and_sorting():
    new_h = {
        "z_file.json": "hash_z_new",
        "a_file.json": "hash_a_new",
        "added.json": "hash_added",
    }
    ref_h = {
        "z_file.json": "hash_z_old",
        "a_file.json": "hash_a_old",
        "removed.json": "hash_removed",
    }

    mismatches = compare_hashes(new_h, ref_h)
    expected = [
        "a_file.json (Old: hash_a_old -> New: hash_a_new)",
        "added.json (Old: None -> New: hash_added)",
        "removed.json (Old: hash_removed -> New: None)",
        "z_file.json (Old: hash_z_old -> New: hash_z_new)",
    ]
    assert mismatches == expected


def test_get_parser():
    parser = get_parser()
    args = parser.parse_args([])
    assert str(args.new_hashes) == "-"
    assert args.ref_hashes.name == "registry_validation.txt"

    custom_args = parser.parse_args(["new.txt", "ref.txt"])
    assert custom_args.new_hashes == Path("new.txt")
    assert custom_args.ref_hashes == Path("ref.txt")


def test_main_success(tmp_path: Path, capsys, monkeypatch):
    new_file = tmp_path / "new.txt"
    ref_file = tmp_path / "ref.txt"
    new_file.write_text("file1.json hash1\n")
    ref_file.write_text("file1.json hash1\n")

    monkeypatch.setattr(
        sys,
        "argv",
        ["check_hashes.py", str(new_file), str(ref_file)],
    )
    main()
    captured = capsys.readouterr()
    assert "Success" in captured.out


def test_main_mismatch(tmp_path: Path, capsys, monkeypatch):
    new_file = tmp_path / "new.txt"
    ref_file = tmp_path / "ref.txt"
    new_file.write_text("file1.json hash_new\n")
    ref_file.write_text("file1.json hash_old\n")

    monkeypatch.setattr(
        sys,
        "argv",
        ["check_hashes.py", str(new_file), str(ref_file)],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "MISMATCH DETECTED" in captured.out


def test_main_missing_file(tmp_path: Path, capsys, monkeypatch):
    missing_file = tmp_path / "missing.txt"
    ref_file = tmp_path / "ref.txt"
    ref_file.write_text("file1.json hash1\n")

    monkeypatch.setattr(
        sys,
        "argv",
        ["check_hashes.py", str(missing_file), str(ref_file)],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()

    assert exc_info.value.code == 1
    captured = capsys.readouterr()
    assert "Error: File not found" in captured.err
