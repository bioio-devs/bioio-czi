#!/usr/bin/env python
# -*- coding: utf-8 -*-

import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np
import pytest
from _aicspylibczi import PylibCZI_CDimCoordinatesOverspecifiedException
from bioio_base import dimensions, exceptions, test_utilities

from bioio_czi import Reader

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


@pytest.mark.xfail(
    raises=exceptions.UnsupportedFileFormatError,
    reason="Do not support remote CZI reading in aicspylibczi mode",
)
def test_czi_reader_remote_xfail() -> None:
    # Construct full filepath
    uri = (
        "https://allencell.s3.amazonaws.com/aics/hipsc_12x_overview_image_dataset/"
        "stitchedwelloverviewimagepath/05080558_3500003720_10X_20191220_D3.czi"
    )
    Reader(uri, use_aicspylibczi=True)


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
