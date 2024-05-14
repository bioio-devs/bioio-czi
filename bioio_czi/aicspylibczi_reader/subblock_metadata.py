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


def _acquisition_time(czi: CziFile, which_subblock: int) -> Optional[np.datetime64]:
    subblock_metadata = czi.read_subblock_metadata(
        Z=0, C=0, T=which_subblock, R=0, S=0, I=0, H=0, V=0
    )
    if not subblock_metadata:
        return None

    metablock_of_first_subblock = subblock_metadata[0][1]
    return _extract_acquisition_time_from_subblock_metadata(metablock_of_first_subblock)


def _difference_in_ms(start_time: np.datetime64, end_time: np.datetime64) -> float:
    delta = end_time - start_time
    # Difference in milliseconds is delta divided by 1 millisecond
    # Explicit conversion to float is required, lest we get a numpy float64
    return float(delta / np.timedelta64(1, "ms"))


def time_between_subblocks(
    czi: CziFile, start_subblock_index: int, end_subblock_index: int
) -> Optional[float]:
    """Calculates the time between two subblocks in milliseconds."""
    start_time = _acquisition_time(czi, start_subblock_index)
    end_time = _acquisition_time(czi, end_subblock_index)
    if start_time is None or end_time is None:
        return None
    return _difference_in_ms(start_time, end_time)


def elapsed_time_all_subblocks(czi: CziFile) -> Optional[float]:
    """
    Looks at all subblocks in the first scene and calculates the time between the first
    and last.

    Parameters
    ----------
    czi : CziFile
        The CziFile object to extract metadata from.

    Returns
    -------
    Optional[float]
        The elapsed time in milliseconds between the first and last subblocks with
        acquisition time metadata, or None if there are fewer than 2 parseable
        subblocks. This may be greater than the number of timepoints times the interval
        between timepoints. This may be nonzero even if there is only one timepoint.

    Example
    -------
    Consider an image with acquisitions at the following times.
        T=0 C=0: 2020-01-01 00:01:00.0000000
        T=0 C=1: 2020-01-01 00:01:00.1234567
        T=1 C=0: 2020-01-01 00:02:00.0000000
        T=1 C=1: 2020-01-01 00:02:00.1234567
    The return value would be 60123.4567 (1 minute and 0.1234567 seconds).
    """
    all_subblocks: list[tuple[dict, str]] = czi.read_subblock_metadata(S=0)
    all_acquisition_times = [
        _extract_acquisition_time_from_subblock_metadata(metadata)
        for _, metadata in all_subblocks
    ]
    # NaT is numpy's representation of "Not a Time", which is used when parsing fails.
    filtered_acquisition_times = [
        t
        for t in all_acquisition_times
        if t is not None and t is not np.datetime64("NaT")
    ]
    sorted_acquisition_times = np.sort(filtered_acquisition_times)
    if len(sorted_acquisition_times) < 2:
        return None
    return _difference_in_ms(sorted_acquisition_times[0], sorted_acquisition_times[-1])
