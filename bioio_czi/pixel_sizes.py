from typing import Optional
from xml.etree import ElementTree as ET

from bioio_base.types import PhysicalPixelSizes

from .metadata import UnsupportedMetadataError


def get_physical_pixel_sizes(metadata: ET.Element) -> PhysicalPixelSizes:
    """
    Look up pixel sizes for X, Y, and Z dimensions from a CZI's XML.
    """
    return PhysicalPixelSizes(
        Z=_single_physical_pixel_size(metadata, "Z", allow_none=True),
        Y=_single_physical_pixel_size(metadata, "Y"),
        X=_single_physical_pixel_size(metadata, "X"),
    )


def _single_physical_pixel_size(
    metadata: ET.Element, dimension: str, allow_none: bool = False
) -> Optional[float]:
    """
    Look up physical pixel size for one dimension.
    """
    scales = metadata.findall(f"./Metadata/Scaling/Items/Distance[@Id='{dimension}']")

    if len(scales) != 1:
        if allow_none and len(scales) == 0:
            return None
        raise UnsupportedMetadataError(
            f"Expected 1 distance scale for dimension '{dimension}' but found "
            f"{len(scales)}."
        )

    unparsed_scale = scales[0].find("./Value")
    if unparsed_scale is None or unparsed_scale.text is None:
        raise UnsupportedMetadataError(
            f"Could not find any distance scale for dimension '{dimension}'."
        )
    scale = float(unparsed_scale.text)
    # The values are stored in units of meters always in .czi. Convert to
    # microns.
    return scale / 1e-6
