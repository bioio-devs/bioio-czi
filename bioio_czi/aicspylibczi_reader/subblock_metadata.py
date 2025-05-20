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


def _acquisition_time(
    czi: CziFile, scene: int, which_subblock: int
) -> Optional[np.datetime64]:
    subblock_metadata = czi.read_subblock_metadata(
        Z=0, C=0, T=which_subblock, R=0, S=scene, I=0, H=0, V=0
    )
    if not subblock_metadata:
        return None

    # subblock_metadata should be a list of length 1 whose only element is a tuple:
    # ({ Z=0, C=0, T=which_subblock, R=0, S=scene, I=0, H=0, V=0 }, metadata_string)
    metablock_of_first_subblock = subblock_metadata[0][1]
    return _extract_acquisition_time_from_subblock_metadata(metablock_of_first_subblock)


def time_between_subblocks(
    czi: CziFile, current_scene: int, start_subblock_index: int, end_subblock_index: int
) -> Optional[float]:
    """Calculates the time between two subblocks in milliseconds."""
    start_time = _acquisition_time(czi, current_scene, start_subblock_index)
    end_time = _acquisition_time(czi, current_scene, end_subblock_index)
    if start_time is None or end_time is None:
        return None
    delta = end_time - start_time
    # Difference in milliseconds is delta divided by 1 millisecond
    # Explicit conversion to float is required, lest we get a numpy float64
    return float(delta / np.timedelta64(1, "ms"))
