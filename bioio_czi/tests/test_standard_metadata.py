import datetime
from typing import Any

import pytest

from bioio_czi import Reader

from .conftest import LOCAL_RESOURCES_DIR


@pytest.mark.parametrize(
    "filename, expected",
    [
        (
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
                "Stage Position X": 32056.045,
                "Stage Position Y": 31179.085,
                "Timelapse": True,
                "Timelapse Interval": datetime.timedelta(milliseconds=59927.0),
                "Total Time Duration": datetime.timedelta(milliseconds=59927.0),
            },
        ),
        (
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
                "Stage Position X": 43832.037,
                "Stage Position Y": 14634.984,
                "Timelapse": False,
                "Timelapse Interval": None,
                "Total Time Duration": None,
            },
        ),
        (
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
    ],
)
def test_standard_metadata(filename: str, expected: dict[str, Any]) -> None:
    uri = LOCAL_RESOURCES_DIR / filename
    reader = Reader(uri)
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
    "filename, scene, expected",
    [
        (
            "variable_per_scene_dims.czi",
            0,
            {
                "Image Size T": 2,
                "Timelapse Interval": datetime.timedelta(milliseconds=59927.0),
                "Total Time Duration": datetime.timedelta(milliseconds=59927.0),
            },
        ),
        (
            "variable_per_scene_dims.czi",
            1,
            {
                "Image Size T": 1,
                "Timelapse Interval": None,
                "Total Time Duration": None,
            },
        ),
    ],
)
def test_standard_metadata_with_set_scene(
    filename: str, scene: int, expected: dict[str, Any]
) -> None:
    # Arrange
    uri = LOCAL_RESOURCES_DIR / filename
    reader = Reader(uri)

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
