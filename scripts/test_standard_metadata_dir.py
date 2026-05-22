"""
Open every .czi file in a target directory using bioio-czi and try to fetch
its standard_metadata. Prints a summary of successes and failures.

Usage:
    python scripts/test_standard_metadata_dir.py [DIRECTORY]

If DIRECTORY is omitted, defaults to the lumenoid assay_optimization path.
"""

import argparse
import sys
import traceback
from pathlib import Path

from bioio_czi import Reader

DEFAULT_DIR = "//allen/aics/lumenoid/assay_optimization/data/3500008658_ZSD0_20260519"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "directory",
        nargs="?",
        default=DEFAULT_DIR,
        help="Directory containing .czi files to test.",
    )
    args = parser.parse_args()

    directory = Path(args.directory)
    if not directory.is_dir():
        print(f"ERROR: not a directory: {directory}", file=sys.stderr)
        return 2

    czi_files = sorted(directory.glob("*.czi"))
    if not czi_files:
        print(f"No .czi files found in {directory}")
        return 0

    print(f"Found {len(czi_files)} .czi file(s) in {directory}\n")

    successes: list[str] = []
    failures: list[tuple[str, str]] = []

    for czi in czi_files:
        print(f"--- {czi.name} ---")
        try:
            reader = Reader(str(czi), use_aicspylibczi=True)
            meta = reader.standard_metadata
            dims = reader.dims
            print(f"  OK: dims = {dims}")
            print(f"  OK: standard_metadata = {meta}")
            successes.append(czi.name)
        except Exception as exc:
            print(f"  FAILED: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            failures.append((czi.name, f"{type(exc).__name__}: {exc}"))
        print()

    print("=" * 60)
    print(f"Summary: {len(successes)} succeeded, {len(failures)} failed")
    if failures:
        print("\nFailures:")
        for name, err in failures:
            print(f"  - {name}: {err}")

    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
