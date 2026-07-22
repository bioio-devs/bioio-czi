import datetime
from typing import Any

import pytest

from bioio_czi import Reader

from .conftest import LOCAL_RESOURCES_DIR


@pytest.mark.parametrize(
    "use_aicspylibczi, filename, expected",
    [
        (
            True,
            "S=2_4x2_T=2=Z=3_CH=2.czi",
            {
                # aicspylibczi: Imaging Datetime from subblock timestamps.
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
                    2021,
                    6,
                    15,
                    6,
                    14,
                    14,
                    369569,
                    tzinfo=datetime.timezone.utc,
                ),
                "Objective": "5x/0.35Air",
                "Pixel Size X": 0.4,
                "Pixel Size Y": 0.4,
                "Pixel Size Z": 1.0,
                "Position Index": None,
                "Reflectors": None,
                "Row": None,
                "Timelapse": True,
                "Timelapse Interval": datetime.timedelta(seconds=19.160193),
                "Total Time Duration": datetime.timedelta(seconds=19.160193),
            },
        ),
        (
            False,
            "S=2_4x2_T=2=Z=3_CH=2.czi",
            {
                # pylibczirw: Imaging Datetime falls back to the file date.
                "Binning": "1x1",
                "Column": None,
                "Dimensions Present": "TCZYX",
                "Image Size C": 2,
                "Image Size T": 2,
                "Image Size X": 947,
                "Image Size Y": 487,
                "Image Size Z": 3,
                "Imaged By": "M1SRH",
                "Imaging Datetime": datetime.datetime(
                    2021,
                    6,
                    15,
                    6,
                    14,
                    13,
                    823569,
                    tzinfo=datetime.timezone.utc,
                ),
                "Objective": "5x/0.35Air",
                "Pixel Size X": 0.4,
                "Pixel Size Y": 0.4,
                "Pixel Size Z": 1.0,
                "Position Index": None,
                "Reflectors": None,
                "Row": None,
                "Timelapse": True,
                "Timelapse Interval": None,
                "Total Time Duration": None,
            },
        ),
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
                    2020,
                    1,
                    18,
                    0,
                    16,
                    31,
                    246361,
                    tzinfo=datetime.timezone.utc,
                ),
                "Objective": "10x/0.45Air",
                "Pixel Size X": 0.5416666666666666,
                "Pixel Size Y": 0.5416666666666666,
                "Pixel Size Z": 2.23,
                "Position Index": 1,
                "Reflectors": [
                    "38 HE Green Fluorescent Prot",
                    "43 HE DsRed",
                    "49 DAPI",
                    "RQFT 405/488/568/647",
                    "RQFT 405/488/568/647",
                ],
                "Row": "4",
                "Timelapse": True,
                "Timelapse Interval": datetime.timedelta(seconds=59.927),
                "Total Time Duration": datetime.timedelta(seconds=59.927),
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
                    2020,
                    1,
                    18,
                    0,
                    16,
                    29,
                    771361,
                    tzinfo=datetime.timezone.utc,
                ),
                "Objective": "10x/0.45Air",
                "Pixel Size X": 0.5416666666666666,
                "Pixel Size Y": 0.5416666666666666,
                "Pixel Size Z": 2.23,
                "Position Index": 1,
                "Reflectors": [
                    "38 HE Green Fluorescent Prot",
                    "43 HE DsRed",
                    "49 DAPI",
                    "RQFT 405/488/568/647",
                    "RQFT 405/488/568/647",
                ],
                "Row": "4",
                "Timelapse": True,
                "Timelapse Interval": None,
                "Total Time Duration": None,
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
                    2016,
                    3,
                    11,
                    10,
                    23,
                    47,
                    745316,
                    tzinfo=datetime.timezone.utc,
                ),
                "Objective": "5x/0.35Air",
                "Pixel Size X": 4.5743626119409,
                "Pixel Size Y": 4.5743626119409,
                "Pixel Size Z": None,
                "Position Index": None,
                "Reflectors": None,
                "Row": None,
                "Timelapse": False,
                "Timelapse Interval": None,
                "Total Time Duration": None,
            },
        ),
        (
            False,
            "OverViewScan.czi",
            {
                # Mosaic stitched in pylibczirw mode (larger X/Y, CYX vs CMYX).
                "Binning": "Other",
                "Column": None,
                "Dimensions Present": "CYX",
                "Image Size C": 1,
                "Image Size T": None,
                "Image Size X": 7398,
                "Image Size Y": 3212,
                "Image Size Z": None,
                "Imaged By": "M1SRH",
                "Imaging Datetime": datetime.datetime(
                    2016,
                    3,
                    11,
                    10,
                    23,
                    44,
                    925154,
                    tzinfo=datetime.timezone.utc,
                ),
                "Objective": "5x/0.35Air",
                "Pixel Size X": 4.5743626119409,
                "Pixel Size Y": 4.5743626119409,
                "Pixel Size Z": None,
                "Position Index": None,
                "Reflectors": None,
                "Row": None,
                "Timelapse": False,
                "Timelapse Interval": None,
                "Total Time Duration": None,
            },
        ),
        (
            False,
            "s_3_t_1_c_3_z_5.czi",
            {
                # Multi-light-source, confocal-style acquisition.
                "Binning": "4x4",
                "Column": None,
                "Dimensions Present": "CZYX",
                "Image Size C": 3,
                "Image Size T": None,
                "Image Size X": 475,
                "Image Size Y": 325,
                "Image Size Z": 5,
                "Imaged By": "ruiany",
                "Imaging Datetime": datetime.datetime(
                    2019,
                    6,
                    27,
                    18,
                    39,
                    25,
                    807886,
                    tzinfo=datetime.timezone.utc,
                ),
                "Objective": "20x/0.8Air",
                "Pixel Size X": 1.0833333333333333,
                "Pixel Size Y": 1.0833333333333333,
                "Pixel Size Z": 1.0,
                "Position Index": 2,
                "Reflectors": [
                    "49 DAPI",
                    "38 HE Green Fluorescent Prot",
                    "43 HE DsRed",
                    "RQFT 405/488/568/647",
                    "RQFT 405/488/568/647",
                ],
                "Row": None,
                "Timelapse": False,
                "Timelapse Interval": None,
                "Total Time Duration": None,
            },
        ),
        (
            False,
            "variable_scene_shape_first_scene_pyramid.czi",
            {
                # Includes a phase-contrast channel.
                "Binning": "1x1",
                "Column": "1",
                "Dimensions Present": "CYX",
                "Image Size C": 3,
                "Image Size T": None,
                "Image Size X": 7705,
                "Image Size Y": 6183,
                "Image Size Z": None,
                "Imaged By": "zeiss",
                "Imaging Datetime": datetime.datetime(
                    2019,
                    5,
                    9,
                    9,
                    49,
                    2,
                    553466,
                    tzinfo=datetime.timezone.utc,
                ),
                "Objective": "5x/0.35Air",
                "Pixel Size X": 0.908210704883533,
                "Pixel Size Y": 0.908210704883533,
                "Pixel Size Z": None,
                "Position Index": 1,
                "Reflectors": None,
                "Row": "1",
                "Timelapse": False,
                "Timelapse Interval": None,
                "Total Time Duration": None,
            },
        ),
        (
            False,
            "w96_A1+A2.czi",
            {
                "Binning": "1x1",
                "Column": "1",
                "Dimensions Present": "CYX",
                "Image Size C": 2,
                "Image Size T": None,
                "Image Size X": 1960,
                "Image Size Y": 1416,
                "Image Size Z": None,
                "Imaged By": "M1SRH",
                "Imaging Datetime": datetime.datetime(
                    2016,
                    7,
                    4,
                    14,
                    52,
                    8,
                    447341,
                    tzinfo=datetime.timezone.utc,
                ),
                "Objective": "20x/0.95Air",
                "Pixel Size X": 0.45715068250381635,
                "Pixel Size Y": 0.45715068250381635,
                "Pixel Size Z": None,
                "Position Index": 1,
                "Reflectors": None,
                "Row": "1",
                "Timelapse": False,
                "Timelapse Interval": None,
                "Total Time Duration": None,
            },
        ),
        (
            False,
            "ome_bounding_box_discrepant.czi",
            {
                # Confocal PMTs; no exposure recorded.
                "Binning": "Other",
                "Column": None,
                "Dimensions Present": "CZYX",
                "Image Size C": 3,
                "Image Size T": None,
                "Image Size X": 2101,
                "Image Size Y": 2101,
                "Image Size Z": 13,
                "Imaged By": "zeiss",
                "Imaging Datetime": datetime.datetime(
                    2025,
                    11,
                    11,
                    16,
                    15,
                    19,
                    816975,
                    tzinfo=datetime.timezone.utc,
                ),
                "Objective": "40x/1.3Oil",
                "Pixel Size X": 0.07602332222751076,
                "Pixel Size Y": 0.07602332222751076,
                "Pixel Size Z": 0.26,
                "Position Index": None,
                "Reflectors": [
                    "38 HE eGFP",
                    "96 HE BFP",
                    "74 HE GFP / mRFP",
                    "50 Cy 5",
                ],
                "Row": None,
                "Timelapse": False,
                "Timelapse Interval": None,
                "Total Time Duration": None,
            },
        ),
    ],
)
def test_standard_metadata(
    use_aicspylibczi: bool, filename: str, expected: dict[str, Any]
) -> None:
    uri = LOCAL_RESOURCES_DIR / filename
    reader = Reader(uri, use_aicspylibczi=use_aicspylibczi)
    standard = reader.standard_metadata
    metadata = standard.to_dict()

    # Every expected label is present, plus the nested "Channels" key and nothing
    # else -- guards against fields silently appearing or disappearing.
    assert set(metadata) == set(expected) | {"Channels"}

    # The nested per-channel metadata matches the standalone channel accessor.
    assert metadata["Channels"] == [c.to_dict() for c in reader.channel_metadata]

    # Every flat field matches exactly (floats compared approximately).
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


@pytest.mark.parametrize(
    "use_aicspylibczi, filename, expected",
    [
        (use_aicspylibczi, filename, expected)
        for use_aicspylibczi in (True, False)
        for filename, expected in [
            (
                "S=2_4x2_T=2=Z=3_CH=2.czi",
                [
                    {
                        "Channel Id": "Channel:0",
                        "Channel Name": "DAPI",
                        "Track": "Track:1",
                        "Dye Name": "DAPI",
                        "Channel Color": "#FF00A1FF",
                        "Contrast Method": "Fluorescence",
                        "Illumination Wavelength": "370-400",
                        "Scan Direction": "Unidirectional",
                        "Excitation Wavelength": "353",
                        "Emission Wavelength": "465",
                        "Effective NA": "0.125",
                        "Exposure Time": "2000000",
                        "Imaging Device": "Internal",
                        "Camera Adapter": "1x Camera Adapter",
                        "Section Thickness": None,
                        "Light Source Intensity": "9.78 %",
                        "Light Source": "LED1",
                    },
                    {
                        "Channel Id": "Channel:1",
                        "Channel Name": "EGFP",
                        "Track": "Track:2",
                        "Dye Name": "EGFP",
                        "Channel Color": "#FF00FF5B",
                        "Contrast Method": "Fluorescence",
                        "Illumination Wavelength": "450-488",
                        "Scan Direction": "Unidirectional",
                        "Excitation Wavelength": "488",
                        "Emission Wavelength": "509",
                        "Effective NA": "0.125",
                        "Exposure Time": "2000000",
                        "Imaging Device": "Internal",
                        "Camera Adapter": "1x Camera Adapter",
                        "Section Thickness": None,
                        "Light Source Intensity": "9.78 %",
                        "Light Source": "LED2",
                    },
                ],
            ),
            (
                "s_1_t_1_c_1_z_1.czi",
                [
                    {
                        "Channel Id": "Channel:0",
                        "Channel Name": "Bright",
                        "Track": "Track:1",
                        "Dye Name": "TL Brightfield",
                        "Channel Color": "#FFFFFFFF",
                        "Contrast Method": "Brightfield",
                        "Illumination Wavelength": None,
                        "Scan Direction": None,
                        "Excitation Wavelength": None,
                        "Emission Wavelength": None,
                        "Effective NA": None,
                        "Exposure Time": "10000000",
                        "Imaging Device": "Camera 2 (Left)",
                        "Camera Adapter": "1.2x EMCCD Camera Adapter",
                        "Section Thickness": "4.587926982854",
                        "Light Source Intensity": "n/a",
                        "Light Source": "Other Lamp",
                    },
                ],
            ),
            (
                "s_3_t_1_c_3_z_5.czi",
                [
                    {
                        "Channel Id": "Channel:0",
                        "Channel Name": "EGFP",
                        "Track": "Track:1",
                        "Dye Name": "EGFP",
                        "Channel Color": "#FF00FF5B",
                        "Contrast Method": "Fluorescence",
                        "Illumination Wavelength": "487-489",
                        "Scan Direction": None,
                        "Excitation Wavelength": "488",
                        "Emission Wavelength": "509",
                        "Effective NA": None,
                        "Exposure Time": "10000000",
                        "Imaging Device": "Camera 2 (Left)",
                        "Camera Adapter": "1.2x EMCCD Camera Adapter",
                        "Section Thickness": "4.5678523399952",
                        "Light Source Intensity": "5.00 %, n/a",
                        "Light Source": "CSU Laserline 2, Other Lamp",
                    },
                    {
                        "Channel Id": "Channel:1",
                        "Channel Name": "TaRFP",
                        "Track": "Track:2",
                        "Dye Name": "TagRFP",
                        "Channel Color": "#FFFF4500",
                        "Contrast Method": "Fluorescence",
                        "Illumination Wavelength": "560-562",
                        "Scan Direction": None,
                        "Excitation Wavelength": "558",
                        "Emission Wavelength": "583",
                        "Effective NA": None,
                        "Exposure Time": "10000000",
                        "Imaging Device": "Camera 2 (Left)",
                        "Camera Adapter": "1.2x EMCCD Camera Adapter",
                        "Section Thickness": "4.6123367179771",
                        "Light Source Intensity": "50.00 %, n/a",
                        "Light Source": "CSU Laserline 3, Other Lamp",
                    },
                    {
                        "Channel Id": "Channel:2",
                        "Channel Name": "Bright",
                        "Track": "Track:3",
                        "Dye Name": "TL Brightfield",
                        "Channel Color": "#FFFFFFFF",
                        "Contrast Method": "Brightfield",
                        "Illumination Wavelength": None,
                        "Scan Direction": None,
                        "Excitation Wavelength": None,
                        "Emission Wavelength": None,
                        "Effective NA": None,
                        "Exposure Time": "10000000",
                        "Imaging Device": "Camera 2 (Left)",
                        "Camera Adapter": "1.2x EMCCD Camera Adapter",
                        "Section Thickness": "4.587926982854",
                        "Light Source Intensity": "n/a",
                        "Light Source": "Other Lamp",
                    },
                ],
            ),
            (
                "variable_scene_shape_first_scene_pyramid.czi",
                [
                    {
                        "Channel Id": "Channel:0",
                        "Channel Name": "EGFP",
                        "Track": "Track:1",
                        "Dye Name": "EGFP",
                        "Channel Color": "#FF00FF5B",
                        "Contrast Method": "Fluorescence",
                        "Illumination Wavelength": "450-488",
                        "Scan Direction": "Bidirectional",
                        "Excitation Wavelength": "488",
                        "Emission Wavelength": "509",
                        "Effective NA": "0.25",
                        "Exposure Time": "150000000",
                        "Imaging Device": "Axiocam 506",
                        "Camera Adapter": "1x Camera Adapter",
                        "Section Thickness": None,
                        "Light Source Intensity": "50.00 %",
                        "Light Source": "LED-Module 470nm",
                    },
                    {
                        "Channel Id": "Channel:1",
                        "Channel Name": "mCher",
                        "Track": "Track:2",
                        "Dye Name": "mCherry",
                        "Channel Color": "#FFFF0900",
                        "Contrast Method": "Fluorescence",
                        "Illumination Wavelength": "578-605",
                        "Scan Direction": "Bidirectional",
                        "Excitation Wavelength": "587",
                        "Emission Wavelength": "610",
                        "Effective NA": "0.25",
                        "Exposure Time": "250000000",
                        "Imaging Device": "Axiocam 506",
                        "Camera Adapter": "1x Camera Adapter",
                        "Section Thickness": None,
                        "Light Source Intensity": "100.00 %",
                        "Light Source": "LED-Module 590nm",
                    },
                    {
                        "Channel Id": "Channel:2",
                        "Channel Name": "PGC",
                        "Track": "Track:3",
                        "Dye Name": "TL Phase Gradient",
                        "Channel Color": "#FFFFFFFF",
                        "Contrast Method": "Phase",
                        "Illumination Wavelength": None,
                        "Scan Direction": "Bidirectional",
                        "Excitation Wavelength": None,
                        "Emission Wavelength": None,
                        "Effective NA": "0.25",
                        "Exposure Time": "1000000",
                        "Imaging Device": "Axiocam 506",
                        "Camera Adapter": "1x Camera Adapter",
                        "Section Thickness": None,
                        "Light Source Intensity": "5.00 %",
                        "Light Source": "TL LED Lamp",
                    },
                ],
            ),
            (
                "ome_bounding_box_discrepant.czi",
                [
                    {
                        "Channel Id": "Channel:0",
                        "Channel Name": "AF647-T1",
                        "Track": "Track:1",
                        "Dye Name": "Alexa Fluor 647",
                        "Channel Color": "#FFFF0014",
                        "Contrast Method": "Fluorescence",
                        "Illumination Wavelength": None,
                        "Scan Direction": "Bidirectional",
                        "Excitation Wavelength": "653",
                        "Emission Wavelength": "668",
                        "Effective NA": None,
                        "Exposure Time": None,
                        "Imaging Device": "GaAsP-Pmt2",
                        "Camera Adapter": None,
                        "Section Thickness": None,
                        "Light Source Intensity": None,
                        "Light Source": "MTBLSMLaserLine4",
                    },
                    {
                        "Channel Id": "Channel:1",
                        "Channel Name": "AF488-T2",
                        "Track": "Track:2",
                        "Dye Name": "Alexa Fluor 488",
                        "Channel Color": "#FF00FF33",
                        "Contrast Method": "Fluorescence",
                        "Illumination Wavelength": None,
                        "Scan Direction": "Bidirectional",
                        "Excitation Wavelength": "493",
                        "Emission Wavelength": "517",
                        "Effective NA": None,
                        "Exposure Time": None,
                        "Imaging Device": "GaAsP-Pmt1",
                        "Camera Adapter": None,
                        "Section Thickness": None,
                        "Light Source Intensity": None,
                        "Light Source": "MTBLSMLaserLine2",
                    },
                    {
                        "Channel Id": "Channel:2",
                        "Channel Name": "DAPI-T3",
                        "Track": "Track:3",
                        "Dye Name": "DAPI",
                        "Channel Color": "#FF00A1FF",
                        "Contrast Method": "Fluorescence",
                        "Illumination Wavelength": None,
                        "Scan Direction": "Bidirectional",
                        "Excitation Wavelength": "353",
                        "Emission Wavelength": "465",
                        "Effective NA": None,
                        "Exposure Time": None,
                        "Imaging Device": "GaAsP-Pmt1",
                        "Camera Adapter": None,
                        "Section Thickness": None,
                        "Light Source Intensity": None,
                        "Light Source": "MTBLSMLaserLine1",
                    },
                ],
            ),
            (
                "NoSceneNames.czi",
                [
                    {
                        "Channel Id": "306379497610784341352213895586310002703",
                        "Channel Name": "ChS1",
                        "Track": "Track:0",
                        "Dye Name": "Alexa Fluor 647",
                        "Channel Color": None,
                        "Contrast Method": "Fluorescence",
                        "Illumination Wavelength": None,
                        "Scan Direction": None,
                        "Excitation Wavelength": "633",
                        "Emission Wavelength": "672.00227050000001",
                        "Effective NA": None,
                        "Exposure Time": None,
                        "Imaging Device": "Detector:0:0",
                        "Camera Adapter": None,
                        "Section Thickness": None,
                        "Light Source Intensity": None,
                        "Light Source": "LightSource:0",
                    },
                ],
            ),
        ]
    ],
)
def test_channel_metadata(
    use_aicspylibczi: bool, filename: str, expected: list
) -> None:
    uri = LOCAL_RESOURCES_DIR / filename
    reader = Reader(uri, use_aicspylibczi=use_aicspylibczi)

    channels = [c.to_dict() for c in reader.channel_metadata]

    assert channels == expected
