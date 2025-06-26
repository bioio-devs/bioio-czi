# Support use of type Reader inside definition of Reader
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple, Union
from xml.etree import ElementTree

import xarray as xr
from bioio_base.dimensions import Dimensions
from bioio_base.exceptions import UnsupportedFileFormatError
from bioio_base.reader import Reader as BaseReader
from bioio_base.standard_metadata import StandardMetadata
from bioio_base.types import PathLike, PhysicalPixelSizes, TimeInterval
from fsspec import AbstractFileSystem
from ome_types.model import OME

from bioio_czi import standard_metadata
from bioio_czi.aicspylibczi_reader.reader import Reader as AicsPyLibCziReader
from bioio_czi.pylibczirw_reader.reader import Reader as PylibCziReader

from . import metadata


class Reader(BaseReader):
    """
    Wraps the pylibczirw and aicspylibczi APIs to provide a BioIO Reader plugin
    for volumetric Zeiss CZI images.
    """

    # Note: Any public method overridden by PylibCziReader or AicsPyLibCziReader must
    # explicitly be defined here, using self._implementation
    _implementation: PylibCziReader | AicsPyLibCziReader

    # Although _fs is named with an underscore, it is used by tests, so must be exposed
    # from the implementation.
    @property
    def _fs(self) -> AbstractFileSystem:
        return self._implementation._fs

    # Although _path is named with an underscore, it is used by tests, so must be
    # exposed from the implementation.
    @property
    def _path(self) -> str:
        return self._implementation._path

    @_path.setter
    def _path(self, value: str) -> None:
        self._implementation._path = value

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
        errors = []
        try:
            if PylibCziReader._is_supported_image(fs, path, **kwargs):
                return True
        except Exception as e:
            errors.append(str(e))
        try:
            if AicsPyLibCziReader._is_supported_image(fs, path, **kwargs):
                return True
        except Exception as e:
            errors.append(str(e))
        error_message = (", ").join(errors)
        raise UnsupportedFileFormatError(
            reader_name="bioio-czi ", path=path, msg_extra=error_message
        )

    def __init__(
        self, image: PathLike, use_aicspylibczi: bool = False, **kwargs: Any
    ) -> None:
        """
        Parameters
        ----------
        image: types.PathLike
            Path to image file.
        use_aicspylibczi: bool
            Read CZIs with the aicspylibczi library. Use aicspylibczi if you want to
            read individual tiles from a scene. However, aicspylibczi cannot read files
            over the internet. Default: False
        chunk_dims: Union[str, List[str]]
            Ignored unless use_aicspylibczi is True.
            Which dimensions to create chunks for.
            Default: DEFAULT_CHUNK_DIMS
            Note: DimensionNames.SpatialY, DimensionNames.SpatialX, and
            DimensionNames.Samples, will always be added to the list if not present
            during dask array construction.
        include_subblock_metadata: bool
            Ignored unless use_aicspylibczi is True.
            Whether to append metadata from the subblocks to the rest of the embeded
            metadata.
        fs_kwargs: Dict[str, Any]
            Ignored unless use_aicspylibczi is True.
            Any specific keyword arguments to pass to the fsspec-created filesystem.
            Default: {}
        """
        if use_aicspylibczi:
            self._implementation = AicsPyLibCziReader(image, **kwargs)
        else:
            self._implementation = PylibCziReader(image, **kwargs)

    @property
    def scenes(self) -> Tuple[str, ...]:
        """
        Returns
        -------
        scenes: Tuple[str, ...]
            A tuple of valid scene IDs in the file.

        Notes
        -----
        Scene IDs are strings - not a range of integers.

        When iterating over scenes please use:

        >>> for id in image.scenes

        and not:

        >>> for i in range(len(image.scenes))
        """
        return self._implementation.scenes

    def set_scene(self, scene_id: Union[str, int]) -> None:
        """
        Set the operating scene.

        Parameters
        ----------
        scene_id: Union[str, int]
            The scene id (if string) or scene index (if integer)
            to set as the operating scene.

        Raises
        ------
        IndexError
            The provided scene id or index is not found in the available scene id list.
        TypeError
            The provided value wasn't a string (scene id) or integer (scene index).
        """
        # We must reset properties on the top-level reader because some high-level
        # properties like xarray_dask_data are cached at the top level and only
        # indirectly delegated to the implementation.
        self._reset_self()
        # set_scene must be delegated to the implementation because that's where the
        # scene ID is stored.
        self._implementation.set_scene(scene_id)

    @property
    def current_scene(self) -> str:
        return self._implementation.current_scene

    @property
    def current_scene_index(self) -> int:
        return self._implementation.current_scene_index

    def _read_delayed(self) -> xr.DataArray:
        """
        The delayed data array constructor for the image.

        Returns
        -------
        data: xarray.DataArray
            The fully constructed delayed DataArray.

            It is additionally recommended to closely monitor how dask array chunks are
            managed.

        Notes
        -----
        Shape of returned array depends on the value of use_aicspylibczi. If
        use_aicspylibczi is not True, any scenes with multiple tiles will be
        automatically stitched (where tiles overlap, the highest M-index wins).
        """
        return self._implementation._read_delayed()

    def _read_immediate(self) -> xr.DataArray:
        """
        The immediate data array constructor for the image.

        Returns
        -------
        data: xarray.DataArray
            The fully read data array.

        Notes
        -----
        Shape of returned array depends on the value of use_aicspylibczi. If
        use_aicspylibczi is not True, any scenes with multiple tiles will be
        automatically stitched (where tiles overlap, the highest M-index wins).
        """
        return self._implementation._read_immediate()

    def _get_stitched_dask_mosaic(self) -> xr.DataArray:
        """
        Returns
        -------
        mosaic: xarray.DataArray
            The fully stitched together image. Contains all the dimensions of the image
            with the YX expanded to the full mosaic.

        Notes
        -----
        Shape of returned array depends on the value of use_aicspylibczi.
        """
        return self._implementation._get_stitched_dask_mosaic()

    def _get_stitched_mosaic(self) -> xr.DataArray:
        """
        Returns
        -------
        mosaic: numpy.ndarray
            The fully stitched together image. Contains all the dimensions of the image
            with the YX expanded to the full mosaic.

        Notes
        -----
        Shape of returned array depends on the value of use_aicspylibczi.
        """
        return self._implementation._get_stitched_mosaic()

    @property
    def mosaic_xarray_dask_data(self) -> xr.DataArray:
        """
        Returns
        -------
        xarray_dask_data: xarray.DataArray
            The delayed stiched mosaic image and metadata as an annotated data array.
        """
        return self._implementation.mosaic_xarray_dask_data

    @property
    def mosaic_xarray_data(self) -> xr.DataArray:
        """
        Returns
        -------
        xarray_dask_data: xarray.DataArray
            The in-memory stitched mosaic image and metadata as an annotated data array.
        """
        return self._implementation.mosaic_xarray_data

    @property
    def metadata(self) -> ElementTree.Element:
        """
        Returns
        -------
        metadata: xml.etree.ElementTree.Element
            The metadata for the formats supported by the inhereting Reader.

            If the inheriting Reader supports processing the metadata into a more useful
            format / Python object, this will return the result.

            For both the unprocessed and processed metadata from the file, use
            `xarray_dask_data.attrs` which will contain a dictionary with keys:
            `unprocessed` and `processed` that you can then select.

        Caution
        -------
        This method uses the xml.etree.ElementTree.fromstring, which is vulnerable
        to denial of service attacks from malicious input data. To learn more, see:
        https://docs.python.org/3/library/xml.html#xml-vulnerabilities
        """
        return self._implementation.metadata

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
        return metadata.transform_metadata_with_xslt(
            self._implementation.metadata,
            Path(__file__).parent / "czi-to-ome-xslt/xslt/czi-to-ome.xsl",
        )

    @property
    def physical_pixel_sizes(self) -> PhysicalPixelSizes:
        """
        Returns
        -------
        sizes: PhysicalPixelSizes
            Using available metadata, the floats representing physical pixel sizes for
            dimensions Z, Y, and X.

        Notes
        -----
        We currently do not handle unit attachment to these values. Please see the file
        metadata for unit information.
        """
        return self._implementation.physical_pixel_sizes

    @property
    def mosaic_tile_dims(self) -> Dimensions | None:
        """
        Returns
        -------
        tile_dims: Optional[Dimensions]
            The dimensions for each tile in the mosaic image.
            If the image is not a mosaic image, returns None.
        """
        return self._implementation.mosaic_tile_dims

    def get_mosaic_tile_position(
        self,
        mosaic_tile_index: int,
        **kwargs: int,
    ) -> Tuple[int, int]:
        """
        Get the absolute position of the top left point for a single mosaic tile.

        Parameters
        ----------
        mosaic_tile_index: int
            The index for the mosaic tile to retrieve position information for.
        kwargs: int
            The keywords below allow you to specify the dimensions that you wish
            to match. If you under-specify the constraints you can easily
            end up with a massive image stack.
                       Z = 1   # The Z-dimension.
                       C = 2   # The C-dimension ("channel").
                       T = 3   # The T-dimension ("time").

        Returns
        -------
        top: int
            The Y coordinate for the tile position.
        left: int
            The X coordinate for the tile position.

        Raises
        ------
        UnexpectedShapeError
            The image has no mosaic dimension available.

        Notes
        -----
        Defaults T and C dimensions to 0 if present as dimensions in image
        to avoid reading in massive image stack for large files.
        """
        return self._implementation.get_mosaic_tile_position(
            mosaic_tile_index, **kwargs
        )

    def get_mosaic_tile_positions(self, **kwargs: int) -> List[Tuple[int, int]]:
        """
        Get the absolute positions of the top left points for each mosaic tile
        matching the specified dimensions and current scene.

        Parameters
        ----------
        kwargs: int
            The keywords below allow you to specify the dimensions that you wish
            to match. If you under-specify the constraints you can easily
            end up with a massive image stack.
                       Z = 1   # The Z-dimension.
                       C = 2   # The C-dimension ("channel").
                       T = 3   # The T-dimension ("time").

        Returns
        -------
        mosaic_tile_positions: List[Tuple[int, int]]
            List of the Y and X coordinate for the tile positions.

        Raises
        ------
        UnexpectedShapeError
            The image has no mosaic dimension available.
        """
        return self._implementation.get_mosaic_tile_positions(**kwargs)

    @property
    def time_interval(self) -> TimeInterval:
        """
        Extracts the time interval between the first two time points in milliseconds.
        Returns
        -------
        Optional[float]
            Timelapse interval in milliseconds. Returns None if extraction fails.
        """
        return self._implementation.time_interval

    @property
    def standard_metadata(self) -> StandardMetadata:
        """
        Return the standard metadata for this reader, updating specific fields.
        This implementation calls the base reader's standard_metadata property
        via super() and then assigns the new values.
        """
        # 1. Some of the standard metadata can be read from all bioio Readers in the
        # same way, which the following super() call does. For instance,
        # standard_metadata.timelapse_interval is set to self.time_interval.
        metadata = super().standard_metadata

        # 2. Most of the remaining implementation is identical across pylibczirw and
        # aicspylibczi modes, so it is shared here. The self-contained standard_metadata
        # module holds the implementation for extracting these from the metadata.
        metadata.column = standard_metadata.column(
            self.metadata, self.current_scene_index
        )
        metadata.position_index = standard_metadata.position_index(self.current_scene)
        metadata.row = standard_metadata.row(self.metadata, self.current_scene_index)

        # 3. Finally, total_time_duration is mode-specific, as only aicspylibczi mode
        # has access to the necessary subblock metadata.
        metadata.timelapse_interval = self.time_interval
        metadata.total_time_duration = self._implementation.total_time_duration

        return metadata
