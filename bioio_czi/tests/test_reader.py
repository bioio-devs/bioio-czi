from bioio_czi import Reader
from bioio_czi.aicspylibczi_reader.reader import Reader as AicsPyLibCziRwReader
from bioio_czi.pylibczirw_reader.reader import Reader as PylibCziRwReader

from .conftest import LOCAL_RESOURCES_DIR


def test_use_aicspylibczi_true() -> None:
    # Arrange
    uri = LOCAL_RESOURCES_DIR / "S=2_4x2_T=2=Z=3_CH=2.czi"

    # Act
    reader = Reader(uri, use_aicspylibczi=True)

    # Assert
    assert isinstance(reader._implementation, AicsPyLibCziRwReader)


def test_use_aicspylibczi_false() -> None:
    # Arrange
    uri = LOCAL_RESOURCES_DIR / "S=2_4x2_T=2=Z=3_CH=2.czi"

    # Act: pylibczirw by default
    reader = Reader(uri)

    # Assert
    assert isinstance(reader._implementation, PylibCziRwReader)
