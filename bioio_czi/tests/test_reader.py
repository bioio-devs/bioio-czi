from pathlib import Path

import numpy as np
from ome_types import OME
from pylibCZIrw import czi as pyczi

from bioio_czi import Reader
from bioio_czi.aicspylibczi_reader.reader import Reader as AicsPyLibCziRwReader
from bioio_czi.pylibczirw_reader.reader import Reader as PylibCziRwReader

from .conftest import LOCAL_RESOURCES_DIR


def test_use_aicspylibczi_true() -> None:
    # Arrange
    uri = LOCAL_RESOURCES_DIR / "S=2_4x2_T=2=Z=3_CH=2.czi"

    # Act: aicspylibczi by default
    reader = Reader(uri)

    # Assert
    assert isinstance(reader._implementation, AicsPyLibCziRwReader)


def test_use_aicspylibczi_false() -> None:
    # Arrange
    uri = LOCAL_RESOURCES_DIR / "S=2_4x2_T=2=Z=3_CH=2.czi"

    # Act
    reader = Reader(uri, use_aicspylibczi=False)

    # Assert
    assert isinstance(reader._implementation, PylibCziRwReader)


def test_ome_metadata_without_optional_channel_metadata(tmp_path: Path) -> None:
    # Arrange: pylibCZIrw writes no AcquisitionMode or other optional channel
    # metadata (https://github.com/bioio-devs/bioio-czi/issues/50)
    uri = tmp_path / "minimal_metadata.czi"
    plane = np.zeros((16, 8), dtype=np.uint8)
    with pyczi.create_czi(str(uri), exist_ok=True) as czi:
        for c in range(2):
            czi.write(plane, plane={"T": 0, "C": c, "Z": 0}, scene=0, location=(0, 0))

    # Act
    ome = Reader(uri).ome_metadata

    # Assert
    assert isinstance(ome, OME)
    assert len(ome.images[0].pixels.channels) == 2
