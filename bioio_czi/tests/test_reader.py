import pytest

from bioio_czi import Reader
from bioio_czi.aicspylibczi_reader.reader import Reader as AicsPyLibCziRwReader

from .conftest import LOCAL_RESOURCES_DIR


def test_reads_with_aicspylibczi() -> None:
    # Arrange
    uri = LOCAL_RESOURCES_DIR / "S=2_4x2_T=2=Z=3_CH=2.czi"

    # Act
    reader = Reader(uri)

    # Assert
    assert isinstance(reader._implementation, AicsPyLibCziRwReader)


def test_use_aicspylibczi_true_is_deprecated() -> None:
    # Arrange
    uri = LOCAL_RESOURCES_DIR / "S=2_4x2_T=2=Z=3_CH=2.czi"

    # Act / Assert: still works, but warns that the kwarg has no effect.
    with pytest.warns(DeprecationWarning):
        reader = Reader(uri, use_aicspylibczi=True)

    assert isinstance(reader._implementation, AicsPyLibCziRwReader)


def test_use_aicspylibczi_false_raises() -> None:
    # Arrange
    uri = LOCAL_RESOURCES_DIR / "S=2_4x2_T=2=Z=3_CH=2.czi"

    # Act / Assert: the pylibczirw backend is gone, so False is unsupported.
    with pytest.raises(ValueError):
        Reader(uri, use_aicspylibczi=False)
