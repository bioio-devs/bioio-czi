import datetime
from typing import Any

import pytest

from bioio_czi import Reader

from .conftest import LOCAL_RESOURCES_DIR


# Test each each of the following files in both aicspylibczi and pylibczirw modes.
#   variable_per_scene_dims.czi
#   OverViewScan.czi
# S=2_4x2_T=2=Z=3_CH=2.czi is only tested in aicspylibczi mode since it is used mainly
# for testing duration and interval, which aren't available in pylibczirw mode.
@pytest.mark.parametrize(
    "use_aicspylibczi, filename, expected",
    [
        (
            True,
            "variable_per_scene_dims.czi",
            {
                "Binning": "1x1",
                "Column": "4",
                "Dimensions Present": "TCZYX",
                "Image Size C": 1,
                "Image Size T": 2,
                "Image Size X": 1848,
                "Image Size Y": 1248,
                "Image Size Z": 2,
                "Imaged By": "sara.carlson",
                "Imaging Datetime": datetime.datetime(
                    2020, 1, 18, 0, 16, 29, 771361, tzinfo=datetime.timezone.utc
                ),
                "Objective": "10x/0.45Air",
                "Pixel Size X": 0.5416666666666666,
                "Pixel Size Y": 0.5416666666666666,
                "Pixel Size Z": 2.23,
                "Position Index": 1,
                "Row": "4",
                "Timelapse": True,
                "Timelapse Interval": datetime.timedelta(milliseconds=59927.0),
                "Total Time Duration": datetime.timedelta(milliseconds=59927.0),
            },
        ),
        (
            True,
            "OverViewScan.czi",
            {
                "Binning": "Other",
                "Column": None,
                "Dimensions Present": "CMYX",
                "Image Size C": 1,
                "Image Size T": None,
                "Image Size X": 544,
                "Image Size Y": 440,
                "Image Size Z": None,
                "Imaged By": "M1SRH",
                "Imaging Datetime": datetime.datetime(
                    2016, 3, 11, 10, 23, 44, 925154, tzinfo=datetime.timezone.utc
                ),
                "Objective": "5x/0.35Air",
                "Pixel Size X": 4.5743626119409,
                "Pixel Size Y": 4.5743626119409,
                "Pixel Size Z": None,
                "Position Index": None,
                "Row": None,
                "Timelapse": False,
                "Timelapse Interval": None,
                "Total Time Duration": None,
            },
        ),
        (
            True,
            "S=2_4x2_T=2=Z=3_CH=2.czi",
            {
                "Binning": "1x1",
                "Column": None,
                "Dimensions Present": "HTCZMYX",
                "Image Size C": 2,
                "Image Size T": 2,
                "Image Size X": 256,
                "Image Size Y": 256,
                "Image Size Z": 3,
                "Imaged By": "M1SRH",
                "Imaging Datetime": datetime.datetime(
                    2021, 6, 15, 6, 14, 13, 823569, tzinfo=datetime.timezone.utc
                ),
                "Objective": "5x/0.35Air",
                "Pixel Size X": 0.4,
                "Pixel Size Y": 0.4,
                "Pixel Size Z": 1.0,
                "Position Index": None,
                "Row": None,
                "Timelapse": True,
                "Timelapse Interval": datetime.timedelta(milliseconds=19160.1933),
                "Total Time Duration": datetime.timedelta(milliseconds=19160.1933),
            },
        ),
        (
            False,
            "variable_per_scene_dims.czi",
            {
                "Binning": "1x1",
                "Column": "4",
                "Dimensions Present": "TCZYX",
                "Image Size C": 1,
                "Image Size T": 2,
                "Image Size X": 1848,
                "Image Size Y": 1248,
                "Image Size Z": 2,
                "Imaged By": "sara.carlson",
                "Imaging Datetime": datetime.datetime(
                    2020, 1, 18, 0, 16, 29, 771361, tzinfo=datetime.timezone.utc
                ),
                "Objective": "10x/0.45Air",
                "Pixel Size X": 0.5416666666666666,
                "Pixel Size Y": 0.5416666666666666,
                "Pixel Size Z": 2.23,
                "Position Index": 1,
                "Row": "4",
                "Timelapse": True,
                "Timelapse Interval": None,  # Available only in aicspylibczi mode
                "Total Time Duration": None,  # Available only in aicspylibczi mode
            },
        ),
        (
            False,
            "OverViewScan.czi",
            {
                "Binning": "Other",
                "Column": None,
                "Dimensions Present": "CYX",  # aicspylibczi mode has CMYX
                "Image Size C": 1,
                "Image Size T": None,
                # This image is larger in X and Y when using pylibczirw mode than
                # aicspylibczi mode because all the tiles are stitched together.
                "Image Size X": 7398,
                "Image Size Y": 3212,
                "Image Size Z": None,
                "Imaged By": "M1SRH",
                "Imaging Datetime": datetime.datetime(
                    2016, 3, 11, 10, 23, 44, 925154, tzinfo=datetime.timezone.utc
                ),
                "Objective": "5x/0.35Air",
                "Pixel Size X": 4.5743626119409,
                "Pixel Size Y": 4.5743626119409,
                "Pixel Size Z": None,
                "Position Index": None,
                "Row": None,
                "Timelapse": False,
                "Timelapse Interval": None,  # Available only in aicspylibczi mode
                "Total Time Duration": None,  # Available only in aicspylibczi mode
            },
        ),
    ],
)
def test_standard_metadata(
    use_aicspylibczi: bool, filename: str, expected: dict[str, Any]
) -> None:
    uri = LOCAL_RESOURCES_DIR / filename
    reader = Reader(uri, use_aicspylibczi=use_aicspylibczi)
    metadata = reader.standard_metadata.to_dict()

    # Compare each key's values.
    for key, expected_value in expected.items():
        error_message = f"{key}: Expected: {expected_value}, Actual: {metadata[key]}"
        if isinstance(expected_value, float):
            assert metadata[key] == pytest.approx(expected_value), error_message
        else:
            assert metadata[key] == expected_value, error_message


# These test cases are specifically to check that standard_metadata reports metadata
# of the user-selected scene.
@pytest.mark.parametrize(
    "use_aicspylibczi, filename, scene, expected",
    [
        (
            True,
            "variable_per_scene_dims.czi",
            0,
            {
                "Image Size T": 2,
                "Timelapse Interval": datetime.timedelta(milliseconds=59927.0),
                "Total Time Duration": datetime.timedelta(milliseconds=59927.0),
            },
        ),
        (
            True,
            "variable_per_scene_dims.czi",
            1,
            {
                "Image Size T": 1,
                "Timelapse Interval": None,
                "Total Time Duration": None,
            },
        ),
        (
            False,
            "variable_per_scene_dims.czi",
            0,
            {
                "Image Size T": 2,
            },
        ),
        (
            False,
            "variable_per_scene_dims.czi",
            1,
            {
                # This should be 1, but pylibczirw assumes all scenes have the same
                # shape, so this is a known defect of pylibczirw mode.
                "Image Size T": 2,
            },
        ),
    ],
)
def test_standard_metadata_with_set_scene(
    use_aicspylibczi: bool, filename: str, scene: int, expected: dict[str, Any]
) -> None:
    # Arrange
    uri = LOCAL_RESOURCES_DIR / filename
    reader = Reader(uri, use_aicspylibczi=use_aicspylibczi)

    # Act
    reader.set_scene(scene)
    metadata = reader.standard_metadata.to_dict()

    # Sanity check
    assert reader.current_scene_index == scene

    # Assert
    # Compare only values mentioned in "expected"
    for key, expected_value in expected.items():
        error_message = f"{key}: Expected: {expected_value}, Actual: {metadata[key]}"
        if isinstance(expected_value, float):
            assert metadata[key] == pytest.approx(expected_value), error_message
        else:
            assert metadata[key] == expected_value, error_message
