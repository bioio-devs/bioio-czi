from lxml import etree
from pylibCZIrw.czi import CziReader


def xml_metadata(file: CziReader) -> etree._Element:
    return etree.fromstring(file.raw_metadata)
