"""
Helper functions extracting metadata from subblocks in aicspylibczi mode.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from aicspylibczi import CziFile
from dateutil import parser
from lxml import etree

log = logging.getLogger(__name__)


def _extract_acquisition_time_from_subblock_metadata(
    subblock_metadata: str,
) -> Optional[datetime]:
    """Extracts acquisition time from subblock metadata."""
    outlxml = etree.fromstring(subblock_metadata)
    acquisition_time_element = outlxml.find(".//AcquisitionTime")

    if acquisition_time_element is not None and acquisition_time_element.text:
        try:
            acquisition_time = parser.isoparse(str(acquisition_time_element.text))
            # Keep acquisition timestamps timezone-aware. If timezone is missing in the
            # source metadata, default to UTC for consistent comparisons.
            if acquisition_time.tzinfo is None:
                acquisition_time = acquisition_time.replace(tzinfo=timezone.utc)
            return acquisition_time
        except Exception as exc:
            log.warning("Failed to extract acquisition time: %s", exc, exc_info=True)

    return None


def _acquisition_time(czi: CziFile, scene: int, frame: int) -> Optional[datetime]:
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
    filtered_acquisition_times = [t for t in acquisition_times if t is not None]

    if len(filtered_acquisition_times) == 0:
        return None
    # One timepoint has many acquisitions (e.g., different channels, different Z
    # positions): the "acquisition time" of the timepoint is the start of the first
    # acquisition.
    return min(filtered_acquisition_times)


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
    return delta.total_seconds() * 1000.0


def acquisition_times(
    czi: CziFile, current_scene: int
) -> Optional[list[dict[str, int | datetime]]]:
    """
    Returns the earliest acquisition time for each mosaic tile at each timepoint.

    Parameters
    ----------
    czi: CziFile
        Open CziFile instance.
    current_scene: int
        Scene index to inspect.

    Returns
    -------
    Optional[list[dict[str, int | datetime]]]:
        A list of dictionaries, each containing subblock info and the corresponding
        acquisition time under the key "acquisition_time". The timezone of the
        acquisition times is preserved if available in the source metadata,
        and defaults to UTC if missing. The timezone may differ from the
        timezone of that for the local microscope, since the CZI file typically saves
        the timestamp in UTC, irrespective of the local timezone of the system
        where the file was created. Returns None if extraction fails.
    """

    try:
        acquisition_times: list[dict[str, int | datetime]] = []

        for subblock_info, subblock_metadata in czi.read_subblock_metadata(
            S=current_scene
        ):
            acquisition_time = _extract_acquisition_time_from_subblock_metadata(
                subblock_metadata
            )
            if acquisition_time is None:
                continue
            d = {**subblock_info, "acquisition_time": acquisition_time}
            acquisition_times.append(d)
        return acquisition_times

    except Exception as exc:
        log.warning("Failed to extract frame acquisition times: %s", exc, exc_info=True)
        return None
