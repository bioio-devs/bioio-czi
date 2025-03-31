#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Any, ContextManager, Dict, Optional, Tuple
from xml.etree import ElementTree

import numpy as np
import xarray as xr
from bioio_base import exceptions
from bioio_base import io as io_utils
from bioio_base import types
from bioio_base.dimensions import DimensionNames, Dimensions
from bioio_base.reader import Reader as BaseReader
from bioio_base.types import PhysicalPixelSizes
from fsspec.spec import AbstractFileSystem
from pylibCZIrw import czi

from .. import metadata_ome
from ..channels import get_channel_names, size
from ..metadata_ome import Metadata, UnsupportedMetadataError
from ..pixel_sizes import get_physical_pixel_sizes

log = logging.getLogger(__name__)

PIXEL_DICT = {
    "Gray8": np.uint8,
    "Gray16": np.uint16,
    "Gray32": np.uint32,  # Not supported by underlying pylibCZIrw
    "Gray32Float": np.float32,
    "Bgr24": np.uint8,
    "Bgr48": np.uint16,
    "Bgr96Float": np.float32,  # Supported by pylibCZIrw but not tested in this plugin
    "invalid": np.uint8,
}


class Reader(BaseReader):
    """
    Wraps the pylibczirw API to provide the same BioIO Reader plugin for
    volumetric Zeiss CZI images.

    Parameters
    ----------
    image: types.PathLike
        Path to image file to construct Reader for.
    fs_kwargs: Dict[str, Any]
        Any specific keyword arguments to pass down to the fsspec created filesystem.
        Default: {}
    """

    _xarray_dask_data: Optional["xr.DataArray"] = None
    _xarray_data: Optional["xr.DataArray"] = None
    _dims: Optional[Dimensions] = None
    _metadata: Optional[Metadata] = None
    _scenes: Optional[Tuple[str, ...]] = None
    _current_scene_index: int = 0
    _fs: "AbstractFileSystem"
    _path: str

    @staticmethod
    def _is_supported_image(
        fs: AbstractFileSystem,
        path: str,
        **kwargs: Any,
    ) -> bool:
        """
        Check if file is a supported CZI by attempting to open it. This is a

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
        try:
            with open(path):
                return True
        except RuntimeError:
            return False

    def __init__(self, image: types.PathLike, fs_kwargs: Dict[str, Any] = {}) -> None:
        self._fs, self._path = io_utils.pathlike_to_fs(
            image, enforce_exists=True, fs_kwargs=fs_kwargs
        )

        if not self._is_supported_image(self._fs, self._path):
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__, self._path
            )

    @property
    def scenes(self) -> Tuple[str, ...]:
        """
        Returns
        -------
        scenes: Tuple[str, ...]
            A tuple of valid scene ids in the file.

        Notes
        -----
        Scene IDs are strings - not a range of integers.

        When iterating over scenes please use:

        >>> for id in image.scenes

        and not:

        >>> for i in range(len(image.scenes))
        """

        def scene_name(metadata: Metadata, scene_index: int) -> str:
            scene_info = metadata.findall(
                "./Metadata/Information/Image/Dimensions/"
                f"S/Scenes/Scene[@Index='{scene_index}']"
            )
            if len(scene_info) != 1:
                raise UnsupportedMetadataError(
                    f"Expected 1 scene for index '{scene_index}' "
                    "but found {len(scene_info)}."
                )
            scene_name = scene_info[0].attrib["Name"]
            if type(scene_name) != str:
                # Fall back to index if name is raw bytes
                return str(scene_index)
            return scene_name

        if self._scenes is None:
            with open(self._path) as file:
                # Underlying scene IDs are ints
                scene_ids = file.scenes_bounding_rectangle.keys()
                self._scenes = tuple(scene_name(self.metadata, i) for i in scene_ids)
                if len(self._scenes) < 1:
                    # If there are no scenes, use the default scene ID
                    self._scenes = (metadata_ome.generate_ome_image_id(0),)

        return self._scenes

    def _get_coords(
        self, xml: Metadata, scene_index: int, dims_shape: Dict[str, Any]
    ) -> Dict[str, list | np.ndarray]:
        """
        Generate coordinate arrays for channel dimension ("C") and spatial dimensions
        ("X", "Y", and "Z") based on channel names and physical pixel sizes.

        Time coordinates are not handled here.
        Hypothetically, we could get the interval between time points from the metadata
        and generate a time coordinate array.
        """
        coords: Dict[str, list | np.ndarray] = {}

        channel_names = get_channel_names(xml, scene_index, dims_shape)
        if channel_names is not None:
            coords[DimensionNames.Channel] = channel_names

        # Handle Spatial Dimensions
        for dim_name, scale in self.physical_pixel_sizes._asdict().items():
            if scale is not None and dim_name in dims_shape:
                dim_size = size(dims_shape, dim_name)
                coords[dim_name] = Reader._generate_coord_array(0, dim_size, scale)

        return coords

    def _read_delayed(self) -> xr.DataArray:
        raise NotImplementedError("Wait for my next PR")

    def _read_immediate(self) -> xr.DataArray:
        """
        The immediate data array constructor for the image.

        Returns
        -------
        data: xr.DataArray
            The fully read data array.
        """
        return self._read_delayed().compute()

    def _get_stitched_dask_mosaic(self) -> xr.DataArray:
        """
        This reader always stiches the entire image together, as the underlying
        pylibczirw does not support reading individual tiles.

        Returns
        -------
        mosaic: xr.DataArray
            The fully stitched together image. Contains all the dimensions of the image
            with the YX expanded to the full mosaic.
        """
        return self.xarray_dask_data

    def _get_stitched_mosaic(self) -> xr.DataArray:
        """
        This reader always stiches the entire image together, as the underlying
        pylibczirw does not support reading individual tiles.

        Returns
        -------
        mosaic: np.ndarray
            The fully stitched together image. Contains all the dimensions of the image
            with the YX expanded to the full mosaic.
        """
        return self.xarray_data

    @property
    def mosaic_xarray_dask_data(self) -> xr.DataArray:
        """
        This reader always stiches the entire image together, as the underlying
        pylibczirw does not support reading individual tiles.

        Returns
        -------
        xarray_dask_data: xr.DataArray
            The delayed stiched mosaic image and metadata as an annotated data array.
        """
        return self.xarray_dask_data

    @property
    def mosaic_xarray_data(self) -> xr.DataArray:
        """
        This reader always stiches the entire image together, as the underlying
        pylibczirw does not support reading individual tiles.

        Returns
        -------
        xarray_dask_data: xr.DataArray
            The in-memory stitched mosaic image and metadata as an annotated data array.
        """
        return self.xarray_data

    @property
    def metadata(self) -> Metadata:
        """
        Returns
        -------
        metadata: Any
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
        if self._metadata is None:
            with open(self._path) as file:
                self._metadata = ElementTree.fromstring(file.raw_metadata)
        return self._metadata

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
        return get_physical_pixel_sizes(self.metadata)

    @property
    def mosaic_tile_dims(self) -> None:
        """
        Returns
        -------
        tile_dims: None
            Inherited method. Mosaic tiles are not supported by the underlying
            pylibCZIrw.
        """
        return None


def open(filepath: str) -> ContextManager[czi.CziReader]:
    """
    Wrapper around czi.open_czi to provide type hinting that clarifies the result
    is a czi.CziReader
    """
    if filepath.startswith("http") or filepath.startswith("https"):
        return czi.open_czi(filepath, czi.ReaderFileInputTypes.Curl)
    return czi.open_czi(filepath)
