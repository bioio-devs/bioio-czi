# Support use of type Reader inside definition of Reader
from __future__ import annotations

from pathlib import Path
from typing import Any, Tuple

import xarray as xr
from bioio_base.reader import Reader as BaseReader
from fsspec import AbstractFileSystem
from ome_types.model.ome import OME

from . import utils as metadata_utils


class Reader(BaseReader):
    """
    Wraps the pylibczirw and aicspylibczi APIs to provide a BioIO Reader plugin
    for volumetric Zeiss CZI images.
    """

    @staticmethod
    def _is_supported_image(
        fs: AbstractFileSystem,
        path: str,
        **kwargs: Any,
    ) -> bool:
        """
        Check if file is a supported CZI by attempting to open it.

        Parameters
        ----------
        fs: AbstractFileSystem
            The file system to used for reading.
        path: str
            The path to the file to read.
        kwargs: Any
            Any kwargs used for reading and validation of the file.

        Returns
        -------
        supported: bool
            Boolean value indicating if the file is supported by the reader.
        """
        from bioio_czi.aicspylibczi_reader.reader import (
            AicsPyLibCziReader as AicsPyLibCziReader,
        )

        return AicsPyLibCziReader._is_supported_image(fs, path, **kwargs)

    def __new__(cls, *args: Any, **kwargs: Any) -> Reader:
        from bioio_czi.aicspylibczi_reader.reader import (
            AicsPyLibCziReader as AicsPyLibCziReader,
        )

        if cls is Reader:
            if "use_aicspylibczi" in kwargs and kwargs["use_aicspylibczi"]:
                return super().__new__(AicsPyLibCziReader)
            else:
                raise NotImplementedError("Only aicspylibczi reader is implemented")
        if cls is AicsPyLibCziReader:
            return super().__new__(AicsPyLibCziReader)
        else:
            raise NotImplementedError(
                "Bug: subclasses of the CZI reader must be "
                "explicitly registered with the parent Reader."
            )

    @property
    def ome_metadata(self) -> OME:
        """
        Returns
        -------
        metadata: OME
            The original metadata transformed into the OME specfication.
            This likely isn't a complete transformation but is guarenteed to
            be a valid transformation.
        """
        return metadata_utils.transform_metadata_with_xslt(
            self.metadata,
            Path(__file__).parent / "czi-to-ome-xslt/xslt/czi-to-ome.xsl",
        )

    def _read_delayed(self) -> xr.DataArray:
        raise NotImplementedError(
            "Bug: you should automatically get a subclass of this Reader that "
            "implements _read_delayed when you call BioImage(...) or Reader(...)."
        )

    def _read_immediate(self) -> xr.DataArray:
        raise NotImplementedError(
            "Bug: you should automatically get a subclass of this Reader that "
            "implements _read_immediate when you call BioImage(...) or Reader(...)."
        )

    @property
    def scenes(self) -> Tuple[str, ...]:
        raise NotImplementedError(
            "Bug: you should automatically get a subclass of this Reader that "
            "implements scenes when you call BioImage(...) or Reader(...)."
        )
