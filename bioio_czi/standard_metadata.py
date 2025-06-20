import logging
from typing import Optional
from xml.etree.ElementTree import Element

log = logging.getLogger(__name__)


def position_index(scene: str) -> Optional[int]:
    """
    Extracts the numeric position index from the current scene name.
    Returns
    -------
    Optional[int]
        The numeric part of the scene name.
        Returns None if parsing fails.
    """
    try:
        # Use only the first part before a "-" if present
        prefix = scene.split("-")[0]
        return int(prefix[1:])
    except (IndexError, ValueError) as exc:
        log.warning(
            "Failed to parse position index from scene name '%s': %s",
            scene,
            exc,
            exc_info=True,
        )
    except Exception as exc:
        log.warning("Unexpected error parsing position index: %s", exc, exc_info=True)

    return None


def _row_or_column(
    metadata: Element, current_scene_index: int, row_or_column: str
) -> Optional[str]:
    """
    Extracts the well row or index for the current scene.
    Returns
    -------
    Optional[str]
        The column index as a string. Returns None if not found.
    """
    try:
        scenes = metadata.findall(
            "Metadata/Information/Image/Dimensions/S/Scenes/Scene"
        )
        for scene in scenes:
            scene_index = scene.get("Index")
            if scene_index is not None and int(scene_index) == current_scene_index:
                shape = scene.find("Shape")
                if shape is not None:
                    index = shape.find(
                        "RowIndex" if row_or_column == "row" else "ColumnIndex"
                    )
                    if index is not None:
                        return index.text
    except Exception as exc:
        log.warning(
            f"Failed to extract well {row_or_column} index: %s", exc, exc_info=True
        )

    return None


def column(metadata: Element, current_scene_index: int) -> Optional[str]:
    """
    Extracts the well column index for the current scene.
    Returns
    -------
    Optional[str]
        The column index as a string. Returns None if not found.
    """
    return _row_or_column(metadata, current_scene_index, "column")


def row(metadata: Element, current_scene_index: int) -> Optional[str]:
    """
    Extracts the well row index for the current scene.
    Returns
    -------
    Optional[str]
        The row index as a string. Returns None if not found.
    """
    return _row_or_column(metadata, current_scene_index, "row")
