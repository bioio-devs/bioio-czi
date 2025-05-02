import logging
from datetime import datetime
from typing import Optional
from xml.etree.ElementTree import Element
from zoneinfo import ZoneInfo

from ome_types import OME

from bioio_czi.metadata import get_metadata_element

log = logging.getLogger(__name__)

VALID_OBJECTIVES = [
    "63x/1.2W",
    "20x/0.8",
    "40x/1.2W",
    "100x/1.25W",
    "100x/1.46Oil",
    "44.83x/1.0W",
    "5x/0.12",
    "10x/0.45",
]


def binning(ome_metadata: OME) -> Optional[str]:
    """
    Extracts the binning setting from the OME metadata.
    Returns
    -------
    Optional[str]
        The binning setting as a string. Returns None if not found.
    """
    try:
        el = get_metadata_element(
            ome_metadata, "./Image/Pixels/Channel/DetectorSettings"
        )
        if el is not None:
            return el.get("Binning", None)
    except Exception as exc:
        log.warning("Failed to extract Binning setting: %s", exc, exc_info=True)

    return None


def column(metadata: Element, current_scene_index: int) -> Optional[str]:
    """
    Extracts the well column index for the current scene.
    Returns
    -------
    Optional[str]
        The column index as a string. Returns None if not found.
    """
    try:
        scenes = metadata.findall(
            "Metadata/Information/Image/Dimensions/S/Scenes/Scene"
        )
        for scene in scenes:
            index = scene.get("Index")
            if index is not None and int(index) == current_scene_index:
                shape = scene.find("Shape")
                if shape is not None:
                    col = shape.find("ColumnIndex")
                    if col is not None:
                        return col.text
    except Exception as exc:
        log.warning("Failed to extract well column index: %s", exc, exc_info=True)

    return None


def imaged_by(ome_metadata: OME) -> Optional[str]:
    """
    Extracts the name of the experimenter (user who imaged the sample).
    Returns
    -------
    Optional[str]
        The username of the experimenter. Returns None if not found.
    """
    try:
        el = get_metadata_element(ome_metadata, "./Experimenter")
        if el is not None:
            return el.get("UserName", None)
    except Exception as exc:
        log.warning("Failed to extract Imaged By: %s", exc, exc_info=True)

    return None


def imaging_date(ome_metadata: OME) -> Optional[str]:
    """
    Extracts the acquisition date from the OME metadata.
    Returns
    -------
    Optional[str]
        The acquisition date in ISO format (YYYY-MM-DD) adjusted to Pacific Time.
        Returns None if the acquisition date is not found or cannot be parsed.
    """
    try:
        el = get_metadata_element(ome_metadata, "./Image/AcquisitionDate")
        if el is not None and el.text:
            # Convert from ISO 8601 (e.g., "2025-03-31T12:00:00Z") to datetime
            utc_time = datetime.fromisoformat(el.text.replace("Z", "+00:00"))
            pacific_time = utc_time.astimezone(ZoneInfo("America/Los_Angeles"))
            return pacific_time.date().isoformat()
    except ValueError as exc:
        log.warning("Failed to parse Acquisition Date: %s", exc, exc_info=True)
    except Exception as exc:
        log.warning("Failed to extract Acquisition Date: %s", exc, exc_info=True)

    return None


def objective(ome_metadata: OME) -> Optional[str]:
    """
    Extracts the microscope objective details.

    Returns
    -------
    Optional[str]
        The formatted objective magnification and numerical aperture.
        Returns None if not found.
    """
    try:
        el = get_metadata_element(ome_metadata, "./Instrument/Objective")
        if el is not None:
            nominal_magnification = el.get("NominalMagnification")
            lens_na = el.get("LensNA")
            immersion = el.get("Immersion")

            # Determine the immersion suffix.
            immersion_suffix = ""
            if immersion == "Oil":
                immersion_suffix = "Oil"
            elif immersion == "Water":
                immersion_suffix = "W"

            if nominal_magnification is not None and lens_na is not None:
                raw_objective = (
                    f"{round(float(nominal_magnification))}x/"
                    f"{float(lens_na)}{immersion_suffix}"
                )

                # Check if the raw objective matches one of the valid values.
                if raw_objective in VALID_OBJECTIVES:
                    return raw_objective

                # Otherwise, check if roughly raw_objective.
                for valid in VALID_OBJECTIVES:
                    if raw_objective in valid:
                        return valid
    except Exception as exc:
        log.warning("Failed to extract Objective: %s", exc, exc_info=True)

    return None


def position_index(scene: str) -> Optional[int]:
    """
    Extracts the numeric position index from the current scene name.
    Returns
    -------
    Optional[int]
        The numeric part of the scene name.
        Returns None if parsing fails.
    """
    try:
        # Use only the first part before a "-" if present
        prefix = scene.split("-")[0]
        return int(prefix[1:])
    except (IndexError, ValueError) as exc:
        log.warning(
            "Failed to parse position index from scene name '%s': %s",
            scene,
            exc,
            exc_info=True,
        )
    except Exception as exc:
        log.warning("Unexpected error parsing position index: %s", exc, exc_info=True)

    return None


def row(metadata: Element, current_scene_index: int) -> Optional[str]:
    """
    Extracts the well row index for the current scene.
    Returns
    -------
    Optional[str]
        The row index as a string. Returns None if not found.
    """
    try:
        scenes = metadata.findall(
            "Metadata/Information/Image/Dimensions/S/Scenes/Scene"
        )
        for scene in scenes:
            index = scene.get("Index")
            if index is not None and int(index) == current_scene_index:
                shape = scene.find("Shape")
                if shape is not None:
                    row = shape.find("RowIndex")
                    if row is not None:
                        return row.text
    except Exception as exc:
        log.warning("Failed to extract well row index: %s", exc, exc_info=True)

    return None
