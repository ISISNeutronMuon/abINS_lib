"""Check newly generated validation hashes against reference registry."""

import argparse
from pathlib import Path
import sys


def get_parser() -> argparse.ArgumentParser:
    """Construct argument parser for the check_hashes script.

    Returns:
        Configured ArgumentParser instance.
    """
    default_registry = (
        Path(__file__).resolve().parents[3]
        / "src/abinslib/registries/registry_validation.txt"
    )

    parser = argparse.ArgumentParser(
        description="Compare validation data file hashes against reference registry.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "new_hashes",
        type=Path,
        nargs="?",
        default=Path("-"),
        help=(
            "Path to text file containing newly generated hashes, or '-' "
            "to read from stdin."
        ),
    )
    parser.add_argument(
        "ref_hashes",
        type=Path,
        nargs="?",
        default=default_registry,
        help=("Path to text file containing reference hashes."),
    )
    return parser


def read_hashes(file_path: Path) -> dict[str, str]:
    """Read file hashes from a text file or standard input.

    Args:
        file_path: Path to text file or Path("-") to read from stdin.

    Returns:
        Dictionary mapping filename to hash string.
    """
    lines = (
        sys.stdin.readlines()
        if str(file_path) == "-"
        else file_path.read_text(encoding="utf-8").splitlines()
    )

    hashes = {
        words[0]: words[1]
        for line in lines
        if not line.startswith("#") and (words := line.split())
    }

    return hashes


def compare_hashes(
    new_hashes: dict[str, str],
    ref_hashes: dict[str, str],
) -> list[str]:
    """Compare generated hashes against reference hashes.

    Args:
        new_hashes: Dictionary of newly generated hashes.
        ref_hashes: Dictionary of reference hashes.

    Returns:
        List of mismatch description strings sorted by filename.
    """
    return [
        f"{filename} (Old: {old_hash} -> New: {new_hash})"
        for filename in sorted(new_hashes.keys() | ref_hashes.keys())
        if (
            (old_hash := ref_hashes.get(filename))
            != (new_hash := new_hashes.get(filename))
        )
    ]


def main() -> None:
    """Compare newly generated hashes with a reference registry."""
    args = get_parser().parse_args()

    if str(args.new_hashes) != "-" and not args.new_hashes.is_file():
        print(f"Error: File not found: {args.new_hashes}", file=sys.stderr)
        sys.exit(1)

    if not args.ref_hashes.is_file():
        print(f"Error: File not found: {args.ref_hashes}", file=sys.stderr)
        sys.exit(1)

    print(f"Comparing newly generated hashes with {args.ref_hashes}...")

    new_hashes = read_hashes(args.new_hashes)
    ref_hashes = read_hashes(args.ref_hashes)

    mismatches = compare_hashes(new_hashes, ref_hashes)

    if mismatches:
        print(
            "❌ MISMATCH DETECTED: The generated data differs "
            "from the reference registry!"
        )
        for mismatch in mismatches:
            print(f"  - {mismatch}")
        print(
            "If these changes are expected, update registry with "
            "`pixi run hashes > src/abinslib/registries/registry_validation.txt` "
            "locally, and create a new GitHub release to host the files."
        )
        sys.exit(1)

    print("✅ Success: Newly generated data matches the reference hashes.")


if __name__ == "__main__":
    main()
