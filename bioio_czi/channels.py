import logging
from typing import Any, Dict, List, Optional, Tuple
from xml.etree import ElementTree as ET

from bioio_base.dimensions import DimensionNames

from bioio_czi.bounding_box import size

from .metadata import generate_ome_channel_id

log = logging.getLogger(__name__)


def channels_element(xml: ET.Element, scene_index: int) -> Optional[ET.Element]:
    """Return the <Channels> element for the given scene, or None if absent."""
    channel_sets = xml.findall(".//Image/Dimensions/Channels")
    if len(channel_sets) == 0:
        return None
    if scene_index < len(channel_sets):
        return channel_sets[scene_index]
    return channel_sets[0]


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
    # Select the current scene
    img = channels_element(xml, scene_index)
    if img is None:
        return None

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


def element_text(element: Optional[ET.Element]) -> Optional[str]:
    """Return an element's stripped text, or None if empty/absent."""
    if element is None or element.text is None:
        return None
    stripped = element.text.strip()
    return stripped or None


def build_id_name_map(xml: ET.Element, path: str) -> Dict[str, str]:
    """
    Build a mapping of element Id -> Name for the instrument components found at
    ``path`` (e.g. Detectors or LightSources).
    """
    mapping: Dict[str, str] = {}
    for element in xml.findall(path):
        element_id = element.get("Id")
        name = element.get("Name")
        if element_id is not None and name is not None:
            # Names in the raw metadata sometimes carry padding whitespace.
            mapping[element_id] = name.strip()
    return mapping


def build_detector_adapter_map(xml: ET.Element) -> Dict[str, str]:
    """Build a mapping of detector Id -> camera adapter model."""
    mapping: Dict[str, str] = {}
    for detector in xml.findall(".//Instrument/Detectors/Detector"):
        detector_id = detector.get("Id")
        adapter = element_text(detector.find("./Adapter/Manufacturer/Model"))
        if detector_id is not None and adapter is not None:
            mapping[detector_id] = adapter
    return mapping


def build_dye_name_map(xml: ET.Element) -> Dict[str, str]:
    """Build a mapping of channel Id -> dye name from the display settings."""
    mapping: Dict[str, str] = {}
    for channel in xml.findall(".//DisplaySetting/Channels/Channel"):
        channel_id = channel.get("Id")
        dye_name = element_text(channel.find("DyeName"))
        if channel_id is not None and dye_name is not None:
            mapping[channel_id] = dye_name
    return mapping


def build_track_map(xml: ET.Element) -> Dict[str, str]:
    """Build a mapping of channel Id -> track Id from the <Tracks> element."""
    mapping: Dict[str, str] = {}
    for track in xml.findall(".//Tracks/Track"):
        track_id = track.get("Id")
        if track_id is None:
            continue
        for channel_ref in track.findall("./ChannelRefs/ChannelRef"):
            channel_id = channel_ref.get("Id")
            if channel_id is not None:
                mapping[channel_id] = track_id
    return mapping


def scan_direction(xml: ET.Element) -> Optional[str]:
    """
    Return the acquisition-wide scan direction (e.g. "Unidirectional"), or None.
    This is not a per-channel value in the raw metadata.
    """
    for element in xml.iter("ScanDirection"):
        text = element_text(element)
        if text is not None:
            return text
    return None


def illumination_wavelength(channel: ET.Element) -> Optional[str]:
    """Return the illumination wavelength range (preferred) or peak, or None."""
    element = channel.find("IlluminationWavelength")
    if element is None:
        return None
    return element_text(element.find("Ranges")) or element_text(
        element.find("SinglePeak")
    )


def light_sources(
    channel: ET.Element, light_source_names: Dict[str, str]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract the comma-joined light source intensities and names for a channel.
    """
    intensities: List[str] = []
    names: List[str] = []
    for settings in channel.findall("./LightSourcesSettings/LightSourceSettings"):
        intensity = settings.find("Intensity")
        if intensity is not None and intensity.text:
            intensities.append(intensity.text.strip())
        source = settings.find("LightSource")
        source_id = source.get("Id") if source is not None else None
        if source_id is not None:
            names.append(light_source_names.get(source_id, source_id))
    return (
        ", ".join(intensities) if intensities else None,
        ", ".join(names) if names else None,
    )
