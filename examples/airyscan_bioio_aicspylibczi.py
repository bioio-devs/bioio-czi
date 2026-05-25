#!/usr/bin/env python3
"""
Read Airyscan raw detector planes via bioio-czi in aicspylibczi mode.

Related: https://github.com/bioio-devs/bioio-czi/issues/72

Note: BioImage.xarray_dask_data currently drops the H dimension (32 detectors
show up as T=1). Use bioio_czi.Reader directly for Airyscan raw access.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bioio import BioImage
from bioio_czi import Reader

from airyscan_plot import plot_airyscan_planes


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Read Airyscan detectors with bioio-czi (aicspylibczi mode)."
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
        help="Output PNG path (default: <input_stem>_airyscan_bioio.png)",
    )
    args = parser.parse_args()
    path = args.path.expanduser().resolve()
    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else path.with_name(f"{path.stem}_airyscan_bioio.png")
    )

    print(f"Reading: {path}")

    reader = Reader(str(path), use_aicspylibczi=True)
    da = reader.xarray_dask_data

    print(f"Reader dims order: {reader.dims.order}")
    print(f"Reader shape:      {reader.shape}")
    print(f"xarray dims:       {da.dims}")
    print(f"xarray shape:      {da.shape}")
    print(f"Channel names:       {list(da.coords['C'].values)}")

    img = BioImage(str(path), reader=Reader, use_aicspylibczi=True)
    bioio_da = img.xarray_dask_data
    print()
    print("BioImage.xarray_dask_data (H is collapsed — issue #72):")
    print(f"  dims:  {bioio_da.dims}")
    print(f"  shape: {bioio_da.shape}")

    detector_name = str(da.coords["C"].values[0])
    sum_name = str(da.coords["C"].values[1])

    # 32 Airyscan detector images for acquisition channel 0 (e.g. AF568-T2)
    detectors = da.isel(C=0, T=0, Z=0).compute().values
    print()
    print(f"Detectors (C=0, T=0, Z=0): shape={detectors.shape}, dtype={detectors.dtype}")
    print(
        f"  per-detector max: min={detectors.max(axis=(1, 2)).min():.0f}, "
        f"max={detectors.max(axis=(1, 2)).max():.0f}"
    )

    # Airyscan sum channel (e.g. AF568#-T2): only H=0 has data
    sum_channel = da.isel(C=1, T=0, Z=0, H=0).compute().values
    print()
    print(
        f"Sum channel (C=1, T=0, Z=0, H=0): shape={sum_channel.shape}, "
        f"max={float(sum_channel.max()):.0f}"
    )

    assert da.sizes["H"] == 32, f"expected 32 detectors on H, got {da.sizes['H']}"
    assert detectors.shape == (32, da.sizes["Y"], da.sizes["X"])
    assert sum_channel.shape == (da.sizes["Y"], da.sizes["X"])
    assert float(sum_channel.max()) > 0

    plot_airyscan_planes(
        detectors,
        sum_channel,
        detector_label=detector_name,
        sum_label=f"{sum_name} (sum, H=0)",
        output=output,
    )
    print()
    print(f"Wrote plot: {output}")
    print("OK: extracted 32 detector planes and sum channel via Reader.xarray_dask_data")


if __name__ == "__main__":
    main()
