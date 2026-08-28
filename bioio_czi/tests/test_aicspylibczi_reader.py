#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime
import xml.etree.ElementTree as ET
from typing import Any, List, Tuple

import numpy as np
import pytest
from _aicspylibczi import PylibCZI_CDimCoordinatesOverspecifiedException
from bioio_base import dimensions, exceptions, test_utilities
from dateutil import parser

from bioio_czi import Reader
from bioio_czi.aicspylibczi_reader import reader as aicspylibczi_reader
from bioio_czi.aicspylibczi_reader.reader import Reader as AicsPyLibCziReader
from bioio_czi.aicspylibczi_reader.reader import remote_reads_available

from .conftest import LOCAL_RESOURCES_DIR


@pytest.mark.parametrize(
    ["filename", "num_subblocks", "acquistion_time"],
    [
        ("s_1_t_1_c_1_z_1.czi", 1, "2019-06-27T18:33:41.1154211Z"),
        ("s_3_t_1_c_3_z_5.czi", 45, "2019-06-27T18:39:26.6459707Z"),
        (
            "variable_scene_shape_first_scene_pyramid.czi",
            27,
            "2019-05-09T09:49:17.9414649Z",
        ),
        pytest.param(
            "s_1_t_1_c_1_z_1.ome.tiff",
            None,
            None,
            marks=pytest.mark.xfail(raises=exceptions.UnsupportedFileFormatError),
        ),
    ],
)
def test_subblocks(filename: str, num_subblocks: int, acquistion_time: str) -> None:
    reader = Reader(
        LOCAL_RESOURCES_DIR / filename,
        include_subblock_metadata=True,
        use_aicspylibczi=True,
    )

    subblocks = reader.metadata.findall("./Subblocks/Subblock")

    assert len(subblocks) == num_subblocks
    # Ensure one of the elements in the first subblock has expected data
    at_metadata: ET.Element | None = subblocks[0].find(".//AcquisitionTime")
    assert at_metadata is not None and at_metadata.text == acquistion_time


@pytest.mark.parametrize(
    "filename, "
    "set_scene, "
    "expected_scenes, "
    "expected_shape, "
    "expected_dtype, "
    "expected_dims_order, "
    "expected_channel_names, "
    "expected_physical_pixel_sizes",
    [
        (
            "S=2_4x2_T=2=Z=3_CH=2.czi",
            "TR1",
            ("TR1", "TR2"),
            (1, 2, 2, 3, 8, 256, 256),
            np.uint16,
            "HTCZMYX",
            ["DAPI", "EGFP"],
            (1.0, 0.4, 0.4),
        ),
        (
            "s_1_t_1_c_1_z_1.czi",
            "Image:0",
            ("Image:0",),
            (1, 325, 475),
            np.uint16,
            "CYX",
            ["Bright"],
            (None, 1.0833333333333333, 1.0833333333333333),
        ),
        (
            "s_3_t_1_c_3_z_5.czi",
            "P2",
            ("P2", "P3", "P1"),
            (3, 5, 325, 475),
            np.uint16,
            "CZYX",
            [
                "EGFP",
                "TaRFP",
                "Bright",
            ],
            (1.0, 1.0833333333333333, 1.0833333333333333),
        ),
        (
            "s_3_t_1_c_3_z_5.czi",
            "P3",
            ("P2", "P3", "P1"),
            (3, 5, 325, 475),
            np.uint16,
            "CZYX",
            [
                "EGFP",
                "TaRFP",
                "Bright",
            ],
            (1.0, 1.0833333333333333, 1.0833333333333333),
        ),
        (
            "s_3_t_1_c_3_z_5.czi",
            "P1",
            ("P2", "P3", "P1"),
            (3, 5, 325, 475),
            np.uint16,
            "CZYX",
            [
                "EGFP",
                "TaRFP",
                "Bright",
            ],
            (1.0, 1.0833333333333333, 1.0833333333333333),
        ),
        (
            "RGB-8bit.czi",
            "Image:0",
            ("Image:0",),
            (1, 624, 924, 3),
            np.uint8,
            "TYXS",
            None,
            (None, 1.0833333333333333, 1.0833333333333333),
        ),
        (
            "variable_per_scene_dims.czi",
            "P2-D4",
            ("P1-D4", "P2-D4"),
            (1, 1, 2, 1248, 1848),  # different from the first scene
            np.uint16,
            "TCZYX",
            ["CMDRP"],
            (2.23, 0.5416666666666666, 0.5416666666666666),
        ),
        (
            "mosaic_split_plate_scene_index_offset.czi",
            "B7-B7",
            ("B7-B7",),
            (1, 50, 1248, 1848),
            np.uint16,
            "CMYX",
            ["Bright ONLY"],
            (None, 0.5416666666666666, 0.5416666666666666),
        ),
        pytest.param(
            "variable_scene_shape_first_scene_pyramid.czi",
            "A1",
            ("A1", "A2"),
            (3, 9, 2208, 2752),
            np.uint16,
            "CMYX",
            [
                "EGFP",
                "mCher",
                "PGC",
            ],
            (None, 9.082107048835329, 9.082107048835329),
            marks=pytest.mark.xfail(reason="Missing scenes"),
        ),
        pytest.param(
            "example.txt",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.xfail(raises=exceptions.UnsupportedFileFormatError),
        ),
        pytest.param(
            "s_1_t_1_c_1_z_1.ome.tiff",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            marks=pytest.mark.xfail(raises=exceptions.UnsupportedFileFormatError),
        ),
    ],
)
def test_czi_reader(
    filename: str,
    set_scene: str,
    expected_scenes: Tuple[str, ...],
    expected_shape: Tuple[int, ...],
    expected_dtype: np.dtype,
    expected_dims_order: str,
    expected_channel_names: List[str],
    expected_physical_pixel_sizes: Tuple[float, float, float],
) -> None:
    # Construct full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    # Run checks
    test_utilities.run_image_file_checks(
        ImageContainer=Reader,
        image=uri,
        set_scene=set_scene,
        expected_scenes=expected_scenes,
        expected_current_scene=set_scene,
        expected_shape=expected_shape,
        expected_dtype=expected_dtype,
        expected_dims_order=expected_dims_order,
        expected_channel_names=expected_channel_names,
        expected_physical_pixel_sizes=expected_physical_pixel_sizes,
        expected_metadata_type=ET.Element,
        reader_kwargs={"use_aicspylibczi": True},
    )


REMOTE_URL = (
    "https://allencell.s3.amazonaws.com/aics/hipsc_12x_overview_image_dataset/"
    "stitchedwelloverviewimagepath/05080558_3500003720_10X_20191220_D3.czi"
)


@pytest.mark.skipif(
    not remote_reads_available(),
    reason="This aicspylibczi build was compiled without libCZI's curl stream",
)
def test_czi_reader_remote() -> None:
    reader = Reader(REMOTE_URL, use_aicspylibczi=True)

    assert reader.dims.order == "HCYX"
    assert reader.shape == (1, 1, 5684, 5925)
    assert reader.physical_pixel_sizes.X == pytest.approx(1.0833333333333333)
    assert reader.metadata.tag == "ImageDocument"

    # Reads are served by range requests, so asking for a window pulls only the
    # sub-blocks covering it rather than the whole 5684x5925 image.
    window = reader.get_image_data("YX", C=0, Y=slice(0, 32), X=slice(0, 32))
    assert window.shape == (32, 32)
    assert window.dtype == np.uint16

    # The same read through the dask path, which reopens the image inside the graph.
    assert np.array_equal(np.asarray(reader.dask_data[0, 0, :32, :32]), window)


def test_czi_reader_remote_without_curl_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Remote reads are a build-time option in aicspylibczi, so a build without them
    # has to say so rather than fail somewhere inside libCZI.
    monkeypatch.setattr(aicspylibczi_reader, "remote_reads_available", lambda: False)

    with pytest.raises(exceptions.UnsupportedFileFormatError, match="curl stream"):
        Reader(REMOTE_URL, use_aicspylibczi=True)


def _normalize_entries(entries: List[dict[str, Any]]) -> List[dict[str, int | str]]:
    """
    Normalize list entries for stable comparison irrespective of ordering or types.
    """

    def normalize_acquisition_time(value: Any) -> str:
        acquisition_time = value
        if isinstance(value, str):
            acquisition_time = parser.isoparse(value)
        if acquisition_time.tzinfo is None:
            acquisition_time = acquisition_time.replace(tzinfo=datetime.timezone.utc)
        return acquisition_time.isoformat(timespec="microseconds")

    def normalize_entry(entry: dict[str, Any]) -> dict[str, int | str]:
        normalized: dict[str, int | str] = {}
        for key, value in entry.items():
            if key == "acquisition_time":
                normalized[key] = normalize_acquisition_time(value)
            else:
                normalized[key] = int(value)
        return normalized

    return sorted(
        (normalize_entry(entry) for entry in entries),
        key=lambda entry: tuple(sorted(entry.items())),
    )


@pytest.mark.parametrize(
    ["filename", "set_scene", "expected_acquisition_times"],
    [
        (
            "s_1_t_1_c_1_z_1.czi",
            "Image:0",
            [{"B": 0, "C": 0, "acquisition_time": "2019-06-27T18:33:41.115421100"}],
        ),
        (
            "s_3_t_1_c_3_z_5.czi",
            "P2",
            [
                {
                    "B": 0,
                    "C": 0,
                    "S": 0,
                    "Z": 0,
                    "acquisition_time": "2019-06-27T18:39:26.645970700",
                },
                {
                    "B": 0,
                    "C": 0,
                    "S": 0,
                    "Z": 1,
                    "acquisition_time": "2019-06-27T18:39:27.476053700",
                },
                {
                    "B": 0,
                    "C": 0,
                    "S": 0,
                    "Z": 2,
                    "acquisition_time": "2019-06-27T18:39:28.259132000",
                },
                {
                    "B": 0,
                    "C": 0,
                    "S": 0,
                    "Z": 3,
                    "acquisition_time": "2019-06-27T18:39:29.042210300",
                },
                {
                    "B": 0,
                    "C": 0,
                    "S": 0,
                    "Z": 4,
                    "acquisition_time": "2019-06-27T18:39:29.844290500",
                },
                {
                    "B": 0,
                    "C": 1,
                    "S": 0,
                    "Z": 0,
                    "acquisition_time": "2019-06-27T18:39:26.890995200",
                },
                {
                    "B": 0,
                    "C": 1,
                    "S": 0,
                    "Z": 1,
                    "acquisition_time": "2019-06-27T18:39:27.708076900",
                },
                {
                    "B": 0,
                    "C": 1,
                    "S": 0,
                    "Z": 2,
                    "acquisition_time": "2019-06-27T18:39:28.490155100",
                },
                {
                    "B": 0,
                    "C": 1,
                    "S": 0,
                    "Z": 3,
                    "acquisition_time": "2019-06-27T18:39:29.273233400",
                },
                {
                    "B": 0,
                    "C": 1,
                    "S": 0,
                    "Z": 4,
                    "acquisition_time": "2019-06-27T18:39:30.073313400",
                },
                {
                    "B": 0,
                    "C": 2,
                    "S": 0,
                    "Z": 0,
                    "acquisition_time": "2019-06-27T18:39:27.116017700",
                },
                {
                    "B": 0,
                    "C": 2,
                    "S": 0,
                    "Z": 1,
                    "acquisition_time": "2019-06-27T18:39:27.898095900",
                },
                {
                    "B": 0,
                    "C": 2,
                    "S": 0,
                    "Z": 2,
                    "acquisition_time": "2019-06-27T18:39:28.681174200",
                },
                {
                    "B": 0,
                    "C": 2,
                    "S": 0,
                    "Z": 3,
                    "acquisition_time": "2019-06-27T18:39:29.483254400",
                },
                {
                    "B": 0,
                    "C": 2,
                    "S": 0,
                    "Z": 4,
                    "acquisition_time": "2019-06-27T18:39:30.265332600",
                },
            ],
        ),
        (
            "s_3_t_1_c_3_z_5.czi",
            "P3",
            [
                {
                    "B": 0,
                    "C": 0,
                    "S": 1,
                    "Z": 0,
                    "acquisition_time": "2019-06-27T18:39:31.052411300",
                },
                {
                    "B": 0,
                    "C": 0,
                    "S": 1,
                    "Z": 1,
                    "acquisition_time": "2019-06-27T18:39:31.860492100",
                },
                {
                    "B": 0,
                    "C": 0,
                    "S": 1,
                    "Z": 2,
                    "acquisition_time": "2019-06-27T18:39:32.674573500",
                },
                {
                    "B": 0,
                    "C": 0,
                    "S": 1,
                    "Z": 3,
                    "acquisition_time": "2019-06-27T18:39:33.474653500",
                },
                {
                    "B": 0,
                    "C": 0,
                    "S": 1,
                    "Z": 4,
                    "acquisition_time": "2019-06-27T18:39:34.255731600",
                },
                {
                    "B": 0,
                    "C": 1,
                    "S": 1,
                    "Z": 0,
                    "acquisition_time": "2019-06-27T18:39:31.290435100",
                },
                {
                    "B": 0,
                    "C": 1,
                    "S": 1,
                    "Z": 1,
                    "acquisition_time": "2019-06-27T18:39:32.104516500",
                },
                {
                    "B": 0,
                    "C": 1,
                    "S": 1,
                    "Z": 2,
                    "acquisition_time": "2019-06-27T18:39:32.903596400",
                },
                {
                    "B": 0,
                    "C": 1,
                    "S": 1,
                    "Z": 3,
                    "acquisition_time": "2019-06-27T18:39:33.703676400",
                },
                {
                    "B": 0,
                    "C": 1,
                    "S": 1,
                    "Z": 4,
                    "acquisition_time": "2019-06-27T18:39:34.485754600",
                },
                {
                    "B": 0,
                    "C": 2,
                    "S": 1,
                    "Z": 0,
                    "acquisition_time": "2019-06-27T18:39:31.496455700",
                },
                {
                    "B": 0,
                    "C": 2,
                    "S": 1,
                    "Z": 1,
                    "acquisition_time": "2019-06-27T18:39:32.312537300",
                },
                {
                    "B": 0,
                    "C": 2,
                    "S": 1,
                    "Z": 2,
                    "acquisition_time": "2019-06-27T18:39:33.112617300",
                },
                {
                    "B": 0,
                    "C": 2,
                    "S": 1,
                    "Z": 3,
                    "acquisition_time": "2019-06-27T18:39:33.895695600",
                },
                {
                    "B": 0,
                    "C": 2,
                    "S": 1,
                    "Z": 4,
                    "acquisition_time": "2019-06-27T18:39:34.678773900",
                },
            ],
        ),
        ("RGB-8bit.czi", "Image:0", None),
        (
            "variable_per_scene_dims.czi",
            "P2-D4",
            [
                {
                    "B": 0,
                    "C": 0,
                    "S": 1,
                    "T": 0,
                    "Z": 0,
                    "acquisition_time": "2020-01-18T00:16:32.274361400",
                },
                {
                    "B": 0,
                    "C": 0,
                    "S": 1,
                    "T": 0,
                    "Z": 1,
                    "acquisition_time": "2020-01-18T00:16:32.617361400",
                },
            ],
        ),
    ],
)
def test_frame_acquisition_times_match_expected_values(
    filename: str,
    set_scene: str,
    expected_acquisition_times: List[dict[str, Any]] | None,
) -> None:
    uri = LOCAL_RESOURCES_DIR / filename
    reader = Reader(uri, use_aicspylibczi=True)
    reader.set_scene(set_scene)

    acquisition_times = reader.acquisition_times

    if expected_acquisition_times is None:
        assert not acquisition_times
        return

    assert acquisition_times is not None
    assert all(
        isinstance(entry["acquisition_time"], datetime.datetime)
        and entry["acquisition_time"].tzinfo is not None
        for entry in acquisition_times
    )

    assert _normalize_entries(acquisition_times) == _normalize_entries(
        expected_acquisition_times
    )


def test_frame_acquisition_times_change_with_scene_selection() -> None:
    uri = LOCAL_RESOURCES_DIR / "s_3_t_1_c_3_z_5.czi"
    reader = Reader(uri, use_aicspylibczi=True)

    acquisition_times_by_scene = {}
    for scene in ("P2", "P3", "P1"):
        reader.set_scene(scene)
        acquisition_times = reader.acquisition_times
        assert acquisition_times is not None
        normalized = _normalize_entries(acquisition_times)
        acquisition_times_by_scene[scene] = tuple(
            tuple(sorted(entry.items())) for entry in normalized
        )

    # All scenes in this file should have distinct acquisition timelines.
    assert len(set(acquisition_times_by_scene.values())) == 3


# @pytest.mark.parametrize(
#     "tiles_filename, stitched_filename, tiles_set_scene, stitched_set_scene",
#     [
#         (
#             "OverViewScan.czi",
#             "OverView.npy",
#             "TR1",
#             "Image:0",
#         )
#     ],
# )

# TODO: Should this depend on the ArrayLike plugin when it comes out?

# def test_czi_reader_mosaic_stitching(
#     tiles_filename: str,
#     stitched_filename: str,
#     tiles_set_scene: str,
#     stitched_set_scene: str,
# ) -> None:
#     # Construct full filepath
#     tiles_uri = get_resource_full_path(tiles_filename, LOCAL)
#     stitched_uri = get_resource_full_path(stitched_filename, LOCAL)

#     # Construct reader
#     tiles_reader = Reader(tiles_uri)
#     stitched_np = np.load(stitched_uri)
#     stitched_reader = ArrayLikeReader(stitched_np)

#     # Run checks
#     run_image_container_mosaic_checks(
#         tiles_image_container=tiles_reader,
#         stitched_image_container=stitched_reader,
#         tiles_set_scene=tiles_set_scene,
#         stitched_set_scene=stitched_set_scene,
#     )


@pytest.mark.parametrize(
    "filename, "
    "set_scene, "
    "expected_tile_dims, "
    "select_tile_index, "
    "expected_tile_top_left",
    [
        (
            "OverViewScan.czi",
            "TR1",
            (440, 544),
            0,
            (0, 0),
        ),
        (
            "OverViewScan.czi",
            "TR1",
            (440, 544),
            50,
            (1188, 4406),
        ),
        (
            "OverViewScan.czi",
            "TR1",
            (440, 544),
            3,
            (0, 1469),
        ),
        (
            "OverViewScan.czi",
            "TR1",
            (440, 544),
            119,
            (2772, 0),
        ),
        pytest.param(
            "OverViewScan.czi",
            "TR1",
            (440, 544),
            999,
            None,
            marks=pytest.mark.xfail(
                raises=PylibCZI_CDimCoordinatesOverspecifiedException
            ),
        ),
        pytest.param(
            "s_1_t_1_c_1_z_1.czi",
            "Image:0",
            None,
            None,
            None,
            # File has no mosaic tiles
            marks=pytest.mark.xfail(raises=AssertionError),
        ),
    ],
)
def test_czi_reader_mosaic_tile_inspection(
    filename: str,
    set_scene: str,
    expected_tile_dims: Tuple[int, int],
    select_tile_index: int,
    expected_tile_top_left: Tuple[int, int],
) -> None:
    # Construct full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    # Construct reader
    reader = Reader(uri, use_aicspylibczi=True)
    reader.set_scene(set_scene)

    # Check basics
    assert reader.mosaic_tile_dims is not None
    assert reader.mosaic_tile_dims.Y == expected_tile_dims[0]
    assert reader.mosaic_tile_dims.X == expected_tile_dims[1]

    # Pull tile info for compare
    tile_y_pos, tile_x_pos = reader.get_mosaic_tile_position(select_tile_index)
    assert tile_y_pos == expected_tile_top_left[0]
    assert tile_x_pos == expected_tile_top_left[1]

    # Pull actual pixel data to compare
    tile_from_m_index = reader.get_image_dask_data(
        reader.dims.order.replace(dimensions.DimensionNames.MosaicTile, ""),
        M=select_tile_index,
    ).compute()

    # Position ops construction to pull using array slicing
    position_ops = []
    for dim in reader.dims.order:
        if dim not in [
            dimensions.DimensionNames.MosaicTile,
            dimensions.DimensionNames.SpatialY,
            dimensions.DimensionNames.SpatialX,
        ]:
            position_ops.append(slice(None))
        if dim is dimensions.DimensionNames.SpatialY:
            position_ops.append(
                slice(
                    tile_y_pos,
                    tile_y_pos + reader.mosaic_tile_dims.Y,
                )
            )
        if dim is dimensions.DimensionNames.SpatialX:
            position_ops.append(
                slice(
                    tile_x_pos,
                    tile_x_pos + reader.mosaic_tile_dims.X,
                )
            )

    tile_from_position = reader.mosaic_dask_data[tuple(position_ops)].compute()

    # Assert all close
    # CZI tiles have about 20% overlap it looks
    # Relative tolerance of 300 is enough to pass
    np.testing.assert_allclose(tile_from_m_index, tile_from_position, rtol=300)


@pytest.mark.parametrize(
    "filename, "
    "expected_tile_y_coords, "
    "expected_tile_x_coords, "
    "expected_mosaic_y_coords, "
    "expected_mosaic_x_coords",
    [
        (
            "OverViewScan.czi",
            np.arange(0, 2012.719549253996, 4.5743626119409),
            np.arange(0, 2488.45326089585, 4.5743626119409),
            np.arange(0, 14692.852709554172, 4.5743626119409),
            np.arange(0, 33836.560240526844, 4.5743626119409),
        ),
        (
            "mosaic_split_plate_scene_index_offset.czi",
            np.arange(0, 1248 * 0.5416666666666666, 0.5416666666666666),
            np.arange(0, 1848 * 0.5416666666666666, 0.5416666666666666),
            np.arange(0, 10233 * 0.5416666666666666, 0.5416666666666666),
            np.arange(0, 10164 * 0.5416666666666666, 0.5416666666666666),
        ),
    ],
)
def test_czi_reader_mosaic_coords(
    filename: str,
    expected_tile_y_coords: np.ndarray,
    expected_tile_x_coords: np.ndarray,
    expected_mosaic_y_coords: np.ndarray,
    expected_mosaic_x_coords: np.ndarray,
) -> None:
    # Construct full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    # Construct reader
    reader = Reader(uri, use_aicspylibczi=True)

    # Check tile y and x min max
    np.testing.assert_array_equal(
        reader.xarray_dask_data.coords[dimensions.DimensionNames.SpatialY].data,
        expected_tile_y_coords,
    )
    np.testing.assert_array_equal(
        reader.xarray_dask_data.coords[dimensions.DimensionNames.SpatialX].data,
        expected_tile_x_coords,
    )

    # Check mosaic y and x min max
    np.testing.assert_array_equal(
        reader.mosaic_xarray_dask_data.coords[dimensions.DimensionNames.SpatialY].data,
        expected_mosaic_y_coords,
    )
    np.testing.assert_array_equal(
        reader.mosaic_xarray_dask_data.coords[dimensions.DimensionNames.SpatialX].data,
        expected_mosaic_x_coords,
    )


def test_czi_reader_mosaic_eager_and_dask_agree() -> None:
    """
    The stitched mosaic is read one way when it is pulled whole and another way when
    it is pulled in chunks, so the two paths have to produce the same pixels.
    """
    reader = Reader(
        LOCAL_RESOURCES_DIR / "OverViewScan.czi",
        use_aicspylibczi=True,
    )

    np.testing.assert_array_equal(reader.mosaic_dask_data.compute(), reader.mosaic_data)


def test_czi_reader_mosaic_is_chunked_by_tile() -> None:
    """
    The stitched mosaic must be a grid of chunks rather than one block. A single
    chunk would make every window depend on every tile, which is what makes windowed
    reads expensive over the network.
    """
    reader = Reader(
        LOCAL_RESOURCES_DIR / "OverViewScan.czi",
        use_aicspylibczi=True,
    )

    chunk_rows, chunk_cols = reader.mosaic_dask_data.chunks[-2:]
    assert len(chunk_rows) > 1
    assert len(chunk_cols) > 1
    # Chunks default to one native tile.
    assert chunk_rows[0] == reader.dims.Y
    assert chunk_cols[0] == reader.dims.X


def test_czi_reader_mosaic_window_reads_one_region(monkeypatch: Any) -> None:
    """
    A window smaller than a tile must cost a single region read, not one read per
    tile in the plane. This is the whole point of chunking the mosaic.
    """
    reader = Reader(
        LOCAL_RESOURCES_DIR / "OverViewScan.czi",
        use_aicspylibczi=True,
    )

    regions = []
    original = AicsPyLibCziReader._read_mosaic_region

    def counting_read(*args: Any, **kwargs: Any) -> np.ndarray:
        regions.append(kwargs.get("region"))
        return original(*args, **kwargs)

    monkeypatch.setattr(
        AicsPyLibCziReader, "_read_mosaic_region", staticmethod(counting_read)
    )

    window = reader.mosaic_dask_data[..., 0:64, 0:64].compute()

    assert window.shape[-2:] == (64, 64)
    assert len(regions) == 1


def test_czi_reader_mosaic_chunk_size_is_configurable() -> None:
    """
    Reading whole mosaics is cheaper with chunks larger than a tile, because tiles
    overlap and a tile-sized grid straddles more sub-blocks than a coarse one.
    """
    reader = Reader(
        LOCAL_RESOURCES_DIR / "OverViewScan.czi",
        use_aicspylibczi=True,
        mosaic_chunk_size=(1024, 1024),
    )

    chunk_rows, chunk_cols = reader.mosaic_dask_data.chunks[-2:]
    assert chunk_rows[0] == 1024
    assert chunk_cols[0] == 1024

    default_reader = Reader(
        LOCAL_RESOURCES_DIR / "OverViewScan.czi",
        use_aicspylibczi=True,
    )
    np.testing.assert_array_equal(
        reader.mosaic_dask_data.compute(), default_reader.mosaic_data
    )


def test_czi_reader_mosaic_rejects_non_mosaic_image() -> None:
    reader = Reader(
        LOCAL_RESOURCES_DIR / "s_1_t_1_c_1_z_1.czi",
        use_aicspylibczi=True,
    )

    with pytest.raises(exceptions.InvalidDimensionOrderingError):
        reader.mosaic_data


# ---------------------------------------------------------------------------
# Fast sub-slice read optimization (get_image_data reads only requested planes)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "filename, order, kwargs",
    [
        ("s_3_t_1_c_3_z_5.czi", "ZYX", {"C": 1}),
        ("s_3_t_1_c_3_z_5.czi", "CZYX", {"C": [0, 2]}),
        ("s_3_t_1_c_3_z_5.czi", "CZYX", {"Z": slice(0, 4, 2)}),
        ("s_3_t_1_c_3_z_5.czi", "CZYX", {"C": (0, -1)}),
        ("RGB-8bit.czi", "YXS", {}),
        ("RGB-8bit-with-non-xy-dims.czi", "ZYXS", {"Z": 0}),
    ],
)
def test_get_image_data_matches_full_slice_aics(
    filename: str, order: str, kwargs: dict
) -> None:
    from bioio_base import transforms

    reader = Reader(
        LOCAL_RESOURCES_DIR / filename, use_aicspylibczi=True
    )._implementation
    expected = transforms.reshape_data(reader.data, reader.dims.order, order, **kwargs)
    actual = reader.get_image_data(order, **kwargs)
    np.testing.assert_array_equal(actual, expected)


def test_get_image_data_reads_only_requested_planes_aics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reader = Reader(
        LOCAL_RESOURCES_DIR / "s_3_t_1_c_3_z_5.czi", use_aicspylibczi=True
    )._implementation
    # s_3_t_1_c_3_z_5: T=1, C=3, Z=5 -> dims CZYX. C=1 over Z=5 -> 5 plane reads.
    assert reader.dims.order == "CZYX"

    calls = {"n": 0}
    real_plane = AicsPyLibCziReader._read_plane

    def counting_plane(czi: Any, scene: int, read_dims: Any = None) -> Any:
        calls["n"] += 1
        return real_plane(czi, scene, read_dims)

    monkeypatch.setattr(AicsPyLibCziReader, "_read_plane", staticmethod(counting_plane))
    reader.get_image_data("ZYX", C=1)
    assert calls["n"] == 5


@pytest.mark.parametrize(
    "filename, order, kwargs",
    [
        # Empty selection along a cullable (non-spatial) dim leaves the read loop
        # with nothing to iterate, exercising the empty-result fallback in
        # _read_indexed. The result must keep its full spatial dimensionality.
        ("s_3_t_1_c_3_z_5.czi", "CZYX", {"C": slice(0, 0)}),
        ("s_3_t_1_c_3_z_5.czi", "CZYX", {"Z": slice(0, 0), "C": 1}),
        # RGB (has a Samples axis) empty selection along the cullable T dim.
        ("RGB-8bit.czi", "TYXS", {"T": slice(0, 0)}),
    ],
)
def test_get_image_data_empty_selection_matches_full_slice_aics(
    filename: str, order: str, kwargs: dict
) -> None:
    from bioio_base import transforms

    reader = Reader(
        LOCAL_RESOURCES_DIR / filename, use_aicspylibczi=True
    )._implementation
    expected = transforms.reshape_data(reader.data, reader.dims.order, order, **kwargs)
    actual = reader.get_image_data(order, **kwargs)

    # An empty selection yields a zero-element array that still carries every
    # requested axis (one of them with length 0), matching the base path.
    assert 0 in actual.shape
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)
