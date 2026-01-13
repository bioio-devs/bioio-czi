"""
Helper functions extracting metadata from subblocks in aicspylibczi mode.
"""

import logging
import warnings
from typing import Optional

import numpy as np
from aicspylibczi import CziFile
from lxml import etree

log = logging.getLogger(__name__)


def _extract_acquisition_time_from_subblock_metadata(
    subblock_metadata: str,
) -> Optional[np.datetime64]:
    """Extracts acquisition time from subblock metadata."""
    outlxml = etree.fromstring(subblock_metadata)
    acquisition_time_element = outlxml.find(".//AcquisitionTime")

    if acquisition_time_element is not None and acquisition_time_element.text:
        try:
            # Parse acquisition time using numpy's datetime64 because it supports high
            # precision time (sub-microsecond). This parsing treats timezone-less dates
            # as UTC, which is fine for computing durations.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    "no explicit representation of timezones available",
                    category=UserWarning,
                )
                return np.datetime64(str(acquisition_time_element.text))
        except Exception as exc:
            log.warning("Failed to extract acquisition time: %s", exc, exc_info=True)

    return None


def _acquisition_time(czi: CziFile, scene: int, frame: int) -> Optional[np.datetime64]:
    """Get the time of the first acquisition in the given scene at the given frame."""
    subblocks_at_t: list[tuple[dict, str]] = czi.read_subblock_metadata(
        T=frame, S=scene
    )

    # subblocks_at_t is a list whose element are tuples of the form:
    # ({ Z=0, C=0, T=which_subblock, R=0, S=scene, I=0, H=0, V=0 }, metadata_string)
    acquisition_times = [
        _extract_acquisition_time_from_subblock_metadata(metadata)
        for _, metadata in subblocks_at_t
    ]
    # NaT is numpy's representation of "Not a Time", which is used when parsing fails.
    filtered_acquisition_times = [
        t for t in acquisition_times if t is not None and t is not np.datetime64("NaT")
    ]

    if len(filtered_acquisition_times) == 0:
        return None
    # One timepoint has many acquisitions (e.g., different channels, different Z
    # positions): the "acquisition time" of the timepoint is the start of the first
    # acquisition.
    return np.min(filtered_acquisition_times)


def time_between_subblocks(
    czi: CziFile, current_scene: int, start_frame: int, end_frame: int
) -> Optional[float]:
    """
    Calculates the time from the first acquisition of start_frame to the first
    acquisition of end_frame in milliseconds. Only the given scene is considered.
    """
    start_time = _acquisition_time(czi, current_scene, start_frame)
    end_time = _acquisition_time(czi, current_scene, end_frame)
    if start_time is None or end_time is None:
        return None
    delta = end_time - start_time
    # Difference in milliseconds is delta divided by 1 millisecond
    # Explicit conversion to float is required, lest we get a numpy float64
    return float(delta / np.timedelta64(1, "ms"))


def frame_acquisition_times(
    czi: CziFile, current_scene: int, mosaic_tiles: int, timepoints: int
) -> Optional[list[list[Optional[np.datetime64]]]]:
    """
    Returns the earliest acquisition time for each mosaic tile at each timepoint.

    Parameters
    ----------
    czi: CziFile
        Open CziFile instance.
    current_scene: int
        Scene index to inspect.
    mosaic_tiles: int
        Number of mosaic tiles (M dimension) for the scene.
    timepoints: int
        Number of timepoints (T dimension) for the scene. Defaults to 1 when the
        T dimension is absent.

    Returns
    -------
    Optional[list[list[Optional[np.datetime64]]]]:
        A nested list where the outer index is the mosaic tile and the inner index
        is the timepoint. Each element is the earliest acquisition time observed
        for that tile and timepoint, or None if it could not be determined.
    """
    # Pre-fill output with None so callers can distinguish missing values.
    times: list[list[Optional[np.datetime64]]] = [
        [None for _ in range(timepoints)] for _ in range(mosaic_tiles)
    ]

    try:
        for subblock_info, subblock_metadata in czi.read_subblock_metadata(
            S=current_scene
        ):
            mosaic_index = subblock_info.get("M", 0)
            time_index = subblock_info.get("T", 0)
            if mosaic_index >= mosaic_tiles or time_index >= timepoints:
                continue

            acquisition_time = _extract_acquisition_time_from_subblock_metadata(
                subblock_metadata
            )
            if acquisition_time is None:
                continue

            existing_time = times[mosaic_index][time_index]
            if existing_time is None or acquisition_time < existing_time:
                times[mosaic_index][time_index] = acquisition_time

    except Exception as exc:
        log.warning("Failed to extract frame acquisition times: %s", exc, exc_info=True)
        return None

    # If all entries are None, return None to indicate extraction failure.
    has_values = any(
        acquisition_time is not None for row in times for acquisition_time in row
    )
    return times if has_values else None
