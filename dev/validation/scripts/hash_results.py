"""Compute SHA256 checksums for validation files."""

import hashlib
from itertools import chain
from pathlib import Path
import sys


def sha256sum(filename: Path) -> str:
    """Compute SHA256 hex digest for a file."""
    with open(filename, "rb") as fd:
        return hashlib.file_digest(fd, "sha256").hexdigest()


def main():
    """Print validation data registry format."""
    validation_dir = Path(__file__).parents[1]

    json_files = sorted(
        chain(
            validation_dir.glob("results/*.json"),
            validation_dir.glob("data/*.json"),
        )
    )

    if not json_files:
        print("Warning: No JSON validation files found", file=sys.stderr)

    print("# Validation data registry")
    for file_path in json_files:
        hash_val = sha256sum(file_path)
        print(f"{file_path.name} {hash_val}")


if __name__ == "__main__":
    main()
