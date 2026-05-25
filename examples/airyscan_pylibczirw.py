#!/usr/bin/env python3
"""
Read Airyscan raw detector planes via pylibCZIrw (default bioio backend).

Airyscan stores 32 individual detectors on the H dimension, not T.
Related: https://github.com/bioio-devs/bioio-czi/issues/72
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from pylibCZIrw import czi as pyczi

sys.path.insert(0, str(Path(__file__).resolve().parent))

from airyscan_plot import plot_airyscan_planes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read Airyscan detectors with pylibCZIrw (H dimension)."
    )
    parser.add_argument(
        "path",
        type=Path,
        help="Path to Airyscan CZI",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output PNG path (default: <input_stem>_airyscan_pylibczirw.png)",
    )
    args = parser.parse_args()
    path = args.path.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else path.with_name(f"{path.stem}_airyscan_pylibczirw.png")
    )

    print(f"Reading: {path}")

    with pyczi.open_czi(str(path)) as cz:
        bbox = cz.total_bounding_box
        print(f"total_bounding_box: {bbox}")

        h_count = bbox["H"][1] - bbox["H"][0]
        c_count = bbox["C"][1] - bbox["C"][0]
        print(f"H planes (detectors): {h_count}")
        print(f"C channels:           {c_count}")

        planes = [
            np.asarray(cz.read(plane={"H": h, "C": 0, "Z": 0})).squeeze()
            for h in range(h_count)
        ]
        detectors = np.stack(planes, axis=0)

        sum_img = np.asarray(cz.read(plane={"H": 0, "C": 1, "Z": 0})).squeeze()

        print()
        print(
            f"Detectors (C=0, Z=0, H=0..{h_count - 1}): "
            f"shape={detectors.shape}, dtype={detectors.dtype}"
        )
        print(
            f"  per-detector max: min={detectors.max(axis=(1, 2)).min()}, "
            f"max={detectors.max(axis=(1, 2)).max()}"
        )

        print()
        print(f"Sum channel (C=1, Z=0, H=0): shape={sum_img.shape}, max={sum_img.max()}")

        nonzero_sum_h = [
            h
            for h in range(h_count)
            if np.asarray(cz.read(plane={"H": h, "C": 1, "Z": 0})).max() > 0
        ]
        print(f"Sum channel non-zero at H indices: {nonzero_sum_h}")

    assert detectors.shape[0] == 32
    assert sum_img.max() > 0
    assert nonzero_sum_h == [0]

    plot_airyscan_planes(
        detectors,
        sum_img,
        detector_label="C=0 (detectors)",
        sum_label="C=1 (sum, H=0)",
        output=output,
    )
    print()
    print(f"Wrote plot: {output}")
    print("OK: extracted 32 detector planes and sum channel via pylibCZIrw")


if __name__ == "__main__":
    main()
