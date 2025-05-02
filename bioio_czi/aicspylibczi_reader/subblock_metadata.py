import datetime
import logging
from typing import Optional

from aicspylibczi import CziFile
from lxml import etree

log = logging.getLogger(__name__)


def extract_acquisition_time_from_subblock_metadata(
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
            acquisition_time_from_czi_xml = acquisition_time_element.text
            acquisition_time_as_date = datetime.datetime.strptime(
                acquisition_time_from_czi_xml.split(".")[0], "%Y-%m-%dT%H:%M:%S"
            )

            formatted_acquisition_time = acquisition_time_as_date + datetime.timedelta(
                microseconds=int(str(acquisition_time_from_czi_xml).split(".")[1][:-1])
                / 1000
            )

            return formatted_acquisition_time
        except Exception as exc:
            log.warning("Failed to extract acquisition time: %s", exc, exc_info=True)

    return None


def acquisition_time(czi: CziFile, which_subblock: int) -> Optional[datetime.datetime]:
    subblock_metadata = czi.read_subblock_metadata(
        Z=0, C=0, T=which_subblock, R=0, S=0, I=0, H=0, V=0
    )
    return extract_acquisition_time_from_subblock_metadata(subblock_metadata)


def time_between_subblocks(
    czi: CziFile, start_subblock_index: int, end_subblock_index: int
) -> Optional[int]:
    start_time = acquisition_time(czi, start_subblock_index)
    end_time = acquisition_time(czi, end_subblock_index)

    if start_time is not None and end_time is not None:
        delta = end_time - start_time
        milliseconds: int = round(delta.total_seconds()) * 1000
        return milliseconds

    return None
