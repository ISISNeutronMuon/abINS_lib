"""Check newly generated validation hashes against reference registry."""

import argparse
from pathlib import Path
import sys


def parse_args(args: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Args:
        args: List of command-line argument strings, or None to use sys.argv.

    Returns:
        Parsed arguments namespace.
    """
    default_registry = (
        Path(__file__).resolve().parents[3]
        / "src"
        / "abinslib"
        / "registries"
        / "registry_validation.txt"
    )
    if not default_registry.is_file():
        default_registry = Path("src/abinslib/registries/registry_validation.txt")

    parser = argparse.ArgumentParser(
        description="Compare validation data file hashes against reference registry."
    )
    parser.add_argument(
        "new_hashes",
        type=Path,
        nargs="?",
        default=Path("-"),
        help=(
            "Path to text file containing newly generated hashes, or '-' "
            "to read from stdin (default: '-')."
        ),
    )
    parser.add_argument(
        "ref_hashes",
        type=Path,
        nargs="?",
        default=default_registry,
        help=(
            "Path to text file containing reference hashes "
            "(default: src/abinslib/registries/registry_validation.txt)."
        ),
    )
    return parser.parse_args(args)


def read_hashes(file_path: Path) -> dict[str, str]:
    """Read file hashes from a text file or standard input.

    Args:
        file_path: Path to text file or Path("-") to read from stdin.

    Returns:
        Dictionary mapping filename to hash string.
    """
    hashes = {}
    if str(file_path) == "-":
        lines = sys.stdin.readlines()
    else:
        with file_path.open("r", encoding="utf-8") as fd:
            lines = fd.readlines()

    for line in lines:
        line_str = line.strip()
        if not line_str or line_str.startswith("#"):
            continue
        parts = line_str.split()
        if len(parts) >= 2:
            hashes[parts[0]] = parts[1]

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
        List of mismatch description strings.
    """
    mismatches = []
    for filename, new_hash in new_hashes.items():
        old_hash = ref_hashes.get(filename)
        if old_hash != new_hash:
            mismatches.append(f"{filename} (Old: {old_hash} -> New: {new_hash})")
    return mismatches


def main(args: list[str] | None = None) -> None:
    """Compare newly generated hashes with a reference registry.

    Args:
        args: Command-line arguments.
    """
    parsed_args = parse_args(args)

    new_hashes_path: Path = parsed_args.new_hashes
    ref_hashes_path: Path = parsed_args.ref_hashes

    if str(new_hashes_path) != "-" and not new_hashes_path.is_file():
        print(f"Error: File not found: {new_hashes_path}", file=sys.stderr)
        sys.exit(1)

    if not ref_hashes_path.is_file():
        print(f"Error: File not found: {ref_hashes_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Comparing newly generated hashes with {ref_hashes_path}...")

    new_hashes = read_hashes(new_hashes_path)
    ref_hashes = read_hashes(ref_hashes_path)

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
    else:
        print("✅ Success: Newly generated data matches the reference hashes.")


if __name__ == "__main__":
    main()
