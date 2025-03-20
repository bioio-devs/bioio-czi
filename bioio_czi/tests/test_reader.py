from bioio_czi import Reader
from bioio_czi.aicspylibczi_reader.reader import AicsPyLibCziReader

from .conftest import LOCAL_RESOURCES_DIR


def test_use_aicspylibczi_true() -> None:
    # Arrange
    uri = LOCAL_RESOURCES_DIR / "S=2_4x2_T=2=Z=3_CH=2.czi"

    # Act
    reader = Reader(uri, use_aicspylibczi=True)

    # Assert
    assert isinstance(reader, AicsPyLibCziReader)
