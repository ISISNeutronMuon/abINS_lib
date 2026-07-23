import hashlib
from pathlib import Path
import sys


def sha256sum(filename):
    h = hashlib.sha256()
    b = bytearray(128 * 1024)
    mv = memoryview(b)
    with open(filename, "rb", buffering=0) as f:
        while n := f.readinto(mv):
            h.update(mv[:n])
    return h.hexdigest()


def main():
    validation_dir = Path(__file__).parent.parent

    files_to_hash = [
        ("results", "ethanol_mantid_isotropic_fundamentals.json"),
        ("results", "ethanol_mantid_almost_isotropic_fundamentals.json"),
        ("results", "ethanol_mantid_second_order.json"),
        ("data", "ethanol_qpoint_phonon_modes.json"),
    ]

    print("# Validation data registry")
    for subdir, filename in files_to_hash:
        file_path = validation_dir / subdir / filename
        if file_path.is_file():
            hash_val = sha256sum(file_path)
            print(f"{filename} {hash_val}")
        else:
            print(f"Warning: {filename} not found in {subdir}/", file=sys.stderr)


if __name__ == "__main__":
    main()
