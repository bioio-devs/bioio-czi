"""
Helper functions extracting metadata from subblocks in aicspylibczi mode.
"""
import datetime
import logging
from typing import Optional

from aicspylibczi import CziFile
from lxml import etree

log = logging.getLogger(__name__)


def _extract_acquisition_time_from_subblock_metadata(
    subblock_metadata: list[tuple[dict, str]]
) -> Optional[datetime.datetime]:
    """Extracts acquisition time from subblock metadata."""
    if not subblock_metadata:
        return None

    metablock_of_subblock = subblock_metadata[0][1]
    outlxml = etree.fromstring(metablock_of_subblock)
    acquisition_time_element = outlxml.find(".//AcquisitionTime")

    if acquisition_time_element is not None and acquisition_time_element.text:
        try:
            return datetime.datetime.fromisoformat(str(acquisition_time_element.text))
        except Exception as exc:
            log.warning("Failed to extract acquisition time: %s", exc, exc_info=True)

    return None


def _acquisition_time(czi: CziFile, which_subblock: int) -> Optional[datetime.datetime]:
    subblock_metadata = czi.read_subblock_metadata(
        Z=0, C=0, T=which_subblock, R=0, S=0, I=0, H=0, V=0
    )
    return _extract_acquisition_time_from_subblock_metadata(subblock_metadata)


def time_between_subblocks(
    czi: CziFile, start_subblock_index: int, end_subblock_index: int
) -> Optional[int]:
    """Calculates the time between two subblocks in milliseconds."""
    start_time = _acquisition_time(czi, start_subblock_index)
    end_time = _acquisition_time(czi, end_subblock_index)

    if start_time is not None and end_time is not None:
        delta = end_time - start_time
        milliseconds: int = round(delta.total_seconds()) * 1000
        return milliseconds

    return None
