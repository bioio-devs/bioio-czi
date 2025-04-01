import logging
from typing import Any, Dict, Optional
from xml.etree import ElementTree as ET

from bioio_base.dimensions import DimensionNames

from bioio_czi.bounding_box import size

from .metadata import generate_ome_channel_id

log = logging.getLogger(__name__)


def get_channel_names(
    xml: ET.Element, scene_index: int, dims_shape: Dict[str, Any]
) -> Optional[list[str]]:
    """
    Get the channel names for the given scene index.

    Parameters
    ----------
    metadata: xml.etree.ElementTree.Element
        The metadata to search for channel names.
    scene_index: int
    """
    # Get all images
    img_sets = xml.findall(".//Image/Dimensions/Channels")

    if len(img_sets) == 0:
        return None

    # Select the current scene
    img = img_sets[0]
    if scene_index < len(img_sets):
        img = img_sets[scene_index]

    # Construct channel name list
    scene_channel_list = []
    channels = img.findall("./Channel")
    number_of_channels_in_data = size(dims_shape, DimensionNames.Channel)

    # There may be more channels in the metadata than in the data
    # if so, we will just use the first N channels and log
    # a warning to the user
    if len(channels) > number_of_channels_in_data:
        log.warning(
            "More channels in metadata than in data "
            f"({len(channels)} vs. {number_of_channels_in_data})"
        )

    for i, channel in enumerate(channels[:number_of_channels_in_data]):
        # Id is required, Name is not.
        # But we prefer to use Name if it is present
        channel_name = channel.attrib.get("Name")
        channel_id = channel.attrib.get("Id")
        if channel_name is None:
            # Idea: we could try to find a channel name from
            # DisplaySetting/Channels/Channel
            channel_name = channel_id
        if channel_name is None:
            # This is actually an error because Id was required by the spec
            channel_name = generate_ome_channel_id(str(scene_index), str(i))

        scene_channel_list.append(channel_name)
    return scene_channel_list
