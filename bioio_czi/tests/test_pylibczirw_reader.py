import xml.etree.ElementTree as ET
from typing import List, Tuple

import numpy as np
import pytest
from bioio_base import dimensions, exceptions, test_utilities

from bioio_czi import Reader

from .conftest import LOCAL_RESOURCES_DIR


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
            (2, 2, 3, 487, 947),
            np.uint16,
            "TCZYX",
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
            (624, 924, 3),  # bioio-czi has (1, 624, 924, 3),
            np.uint8,
            "YXS",  # bioio-czi has "TYXS",
            None,
            (None, 1.0833333333333333, 1.0833333333333333),
        ),
        (
            "variable_per_scene_dims.czi",
            "P2-D4",
            ("P1-D4", "P2-D4"),
            # The dimensions of the second scene are actually (1, 1, 2, 1248, 1848),
            # but pylibczirw doesn't make that metadata available and instead assumes
            # all scenes have the same shape.
            # This is a known defect of pylibczirw mode.
            (2, 1, 2, 1248, 1848),
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
    )


@pytest.mark.parametrize(
    "url, expected_shape",
    [
        (
            "https://allencell.s3.amazonaws.com/aics/hipsc_12x_overview_image_dataset/"
            "stitchedwelloverviewimagepath/05080558_3500003720_10X_20191220_D3.czi"
            "?versionId=_KYMRhRvKxnu727ssMD2_fZD5CmQMNw6",
            (1, 5684, 5925),
        ),
    ],
)
def test_czi_reader_remote(url: str, expected_shape: Tuple[int]) -> None:
    # Construct full filepath
    reader = Reader(url)
    assert reader.shape == expected_shape


@pytest.mark.parametrize(
    "filename, expected_y_coords, expected_x_coords",
    [
        (
            "OverViewScan.czi",
            np.arange(0, 14692.852709554172, 4.5743626119409),
            np.arange(0, 33836.560240526844, 4.5743626119409),
        ),
    ],
)
def test_czi_reader_mosaic_coords(
    filename: str, expected_y_coords: np.ndarray, expected_x_coords: np.ndarray
) -> None:
    # Construct full filepath
    uri = LOCAL_RESOURCES_DIR / filename

    # Construct reader
    reader = Reader(uri)

    # Different from bioio-czi reader: the size of xarray_dask_data is the size of
    # a bounding box around all tiles in the scene, not just a single tile. (at the
    # lowest pyramid level (highest resolution)).
    # (at the lowest pyramid level (highest resolution)
    np.testing.assert_array_equal(
        reader.xarray_dask_data.coords[dimensions.DimensionNames.SpatialY].data,
        expected_y_coords,
    )
    np.testing.assert_array_equal(
        reader.xarray_dask_data.coords[dimensions.DimensionNames.SpatialX].data,
        expected_x_coords,
    )

    # Same as bioio-czi reader: mosaic dimensions are the full size of the lowest
    # pyramid level (highest resolution).
    np.testing.assert_array_equal(
        reader.mosaic_xarray_dask_data.coords[dimensions.DimensionNames.SpatialY].data,
        expected_y_coords,
    )
    np.testing.assert_array_equal(
        reader.mosaic_xarray_dask_data.coords[dimensions.DimensionNames.SpatialX].data,
        expected_x_coords,
    )
