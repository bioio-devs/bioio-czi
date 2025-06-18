import os
from pathlib import Path
from typing import Union
from xml.etree import ElementTree as ET

import lxml.etree
from bioio_base.types import PathLike
from ome_types import OME

OME_NS = {"": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}


class UnsupportedMetadataError(Exception):
    """
    The reader encountered metadata it doesn't know how to handle.
    """


def generate_ome_image_id(image_id: Union[str, int]) -> str:
    """
    Naively generates the standard OME image ID using a provided ID.

    Parameters
    ----------
    image_id: Union[str, int]
        A string or int representing the ID for an image.
        In the context of the usage of this function, this is usually used with the
        index of the scene / image.

    Returns
    -------
    ome_image_id: str
        The OME standard for image IDs.
    """
    return f"Image:{image_id}"


def generate_ome_channel_id(image_id: str, channel_id: Union[str, int]) -> str:
    """
    Naively generates the standard OME channel ID using a provided ID.

    Parameters
    ----------
    image_id: str
        An image id to pull the image specific index from.
        See: `generate_ome_image_id` for more details.
    channel_id: Union[str, int]
        A string or int representing the ID for a channel.
        In the context of the usage of this function, this is usually used with the
        index of the channel.

    Returns
    -------
    ome_channel_id: str
        The OME standard for channel IDs.


    Notes
    -----
    ImageIds are usually: "Image:0", "Image:1", or "Image:N",
    ChannelIds are usually the combination of image index + channel index --
    "Channel:0:0" for the first channel of the first image for example.
    """
    # Remove the prefix 'Image:' to get just the index
    image_index = image_id.replace("Image:", "")
    return f"Channel:{image_index}:{channel_id}"


def generate_ome_instrument_id(instrument_id: Union[str, int]) -> str:
    """
    Naively generates the standard OME instrument ID using a provided ID.

    Parameters
    ----------
    instrument_id: Union[str, int]
        A string or int representing the ID for an instrument.

    Returns
    -------
    ome_instrument_id: str
        The OME standard for instrument IDs.
    """
    return f"Instrument:{instrument_id}"


def generate_ome_detector_id(detector_id: Union[str, int]) -> str:
    """
    Naively generates the standard OME detector ID using a provided ID.

    Parameters
    ----------
    detector_id: Union[str, int]
        A string or int representing the ID for a detector.

    Returns
    -------
    ome_detector_id: str
        The OME standard for detector IDs.
    """
    return f"Detector:{detector_id}"


def transform_metadata_with_xslt(
    tree: ET.Element,
    xslt: PathLike,
) -> OME:
    """
    Given an in-memory metadata Element and a path to an XSLT file, convert
    metadata to OME.

    Parameters
    ----------
    tree: ET.Element
        The metadata tree to convert.
    xslt: PathLike
        Path to the XSLT file.

    Returns
    -------
    ome: OME
        The generated / translated OME metadata.

    Notes
    -----
    This function will briefly update your processes current working directory
    to the directory that stores the XSLT file.
    """
    # Store current process directory
    process_dir = Path().cwd()

    # Make xslt path absolute
    xslt_abs_path = Path(xslt).resolve(strict=True).absolute()

    # Try the transform
    try:
        # We switch directories so that whatever sub-moduled in XSLT
        # main file can have local references to supporting transforms.
        # i.e. the main XSLT file imports a transformers for specific sections
        # of the metadata (camera, experiment, etc.)
        os.chdir(xslt_abs_path.parent)

        # Parse template and generate transform function
        template = lxml.etree.parse(str(xslt_abs_path))
        transform = lxml.etree.XSLT(template)

        # Convert from stdlib ET to lxml ET
        tree_str = ET.tostring(tree)
        lxml_tree = lxml.etree.fromstring(tree_str)
        ome_etree = transform(lxml_tree)

        # Dump generated etree to string and read with ome-types
        ome = OME.from_xml(str(ome_etree))

    # Regardless of error or succeed, move back to original process dir
    finally:
        os.chdir(process_dir)

    return ome
