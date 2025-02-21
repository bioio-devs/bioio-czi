#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
from typing import Any, ContextManager, Dict, List, Optional, Tuple
from xml.etree import ElementTree

import dask.array as da
import numpy as np
import xarray as xr
from bioio_base import exceptions
from bioio_base import io as io_utils
from bioio_base import types
from bioio_base.dimensions import (
    DEFAULT_DIMENSION_ORDER_LIST,
    DimensionNames,
    Dimensions,
)
from bioio_base.reader import Reader as BaseReader
from bioio_base.types import PhysicalPixelSizes
from dask import delayed
from fsspec.spec import AbstractFileSystem
from ome_types.model.ome import OME
from pylibCZIrw import czi

from . import ome as metadata_utils

Metadata = ElementTree.Element

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

CZI_SAMPLES_DIM_CHAR = "A"
CZI_BLOCK_DIM_CHAR = "B"
CZI_SCENE_DIM_CHAR = "S"


###############################################################################

PIXEL_DICT = {
    "gray8": np.uint8,
    "gray16": np.uint16,
    "gray32": np.uint32,
    "gray32float": np.float32,
    "bgr24": np.uint8,
    "bgr48": np.uint16,
    "invalid": np.uint8,
}


def pixel_type_is_array(pixel_type: str) -> bool:
    """
    Is each pixel represented by multiple numbers (i.e., RGB)?

    Parameters
    ----------
    pixel_type: str
        The pixel type to check.

    Returns
    -------
    is_array: bool
        True if the pixel type is an array type.
    """
    return "bgr" in pixel_type


###############################################################################


class Reader(BaseReader):
    """
    Wraps the aicspylibczi API to provide the same BioIO Reader plugin for
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
    # Do not provide default value because
    # they may not need to be used by your reader (i.e. input param is an array)
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
            with open_czi_typed(path):
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
            with open_czi_typed(self._path) as file:
                # Underlying scene IDs are ints
                scene_ids = file.scenes_bounding_rectangle.keys()
                self._scenes = tuple(scene_name(self.metadata, i) for i in scene_ids)

        return self._scenes

    @staticmethod
    def _get_coords_and_physical_px_sizes(
        xml: Metadata, scene_index: int, dims_shape: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], types.PhysicalPixelSizes]:
        # Create coord dict
        coords: Dict[str, Any] = {}

        # Get all images
        img_sets = xml.findall(".//Image/Dimensions/Channels")

        if len(img_sets) != 0:
            # Select the current scene
            img = img_sets[0]
            if scene_index < len(img_sets):
                img = img_sets[scene_index]

            # Construct channel name list
            scene_channel_list = []
            channels = img.findall("./Channel")
            number_of_channels_in_data = dims_shape[DimensionNames.Channel][1]

            # There may be more channels in the metadata than in the data
            # if so, we will just use the first N channels and log
            # a warning to the user
            if len(channels) > number_of_channels_in_data:
                log.warning(
                    "More channels in metadata than in data "
                    f"({len(channels)} vs {number_of_channels_in_data})"
                )

            for i, channel in enumerate(channels[:number_of_channels_in_data]):
                # Id is required, Name is not.
                # But we prefer to use Name if it is present
                channel_name = channel.attrib.get("Name")
                channel_id = channel.attrib.get("Id")
                if channel_name is None:
                    # TODO: the next best guess is to see if there's a Name in
                    # DisplaySetting/Channels/Channel
                    # xpath_str = "./Metadata/DisplaySetting/Channels"
                    # displaysetting_channels = xml.findall(xpath_str)
                    # ds_channels = displaysetting_channels[0].findall("./Channel")
                    # to find matching channel must match on Id attribute or if Id not
                    # present, just on collection index i
                    # If we didn't find a match this way, just use the Id as the name
                    channel_name = channel_id
                if channel_name is None:
                    # This is actually an error because Id was required by the spec
                    channel_name = metadata_utils.generate_ome_channel_id(
                        str(scene_index), str(i)
                    )

                scene_channel_list.append(channel_name)

            # Attach channel names to coords
            coords[DimensionNames.Channel] = scene_channel_list

        # Unpack short info scales
        list_xs = xml.findall(".//Distance[@Id='X']")
        list_ys = xml.findall(".//Distance[@Id='Y']")
        list_zs = xml.findall(".//Distance[@Id='Z']")
        scale_xe = list_xs[0].find("./Value")
        scale_ye = list_ys[0].find("./Value")
        scale_ze = None if len(list_zs) == 0 else list_zs[0].find("./Value")

        # Set default scales
        scale_x = None
        scale_y = None
        scale_z = None

        # Unpack the string value to a float
        # the values are stored in units of meters always in .czi, so
        # divide by 1E-6 to convert to microns
        if scale_xe is not None and scale_xe.text is not None:
            scale_x = float(scale_xe.text) / (1e-6)
        if scale_ye is not None and scale_ye.text is not None:
            scale_y = float(scale_ye.text) / (1e-6)
        if scale_ze is not None and scale_ze.text is not None:
            scale_z = float(scale_ze.text) / (1e-6)

        # Handle Spatial Dimensions
        for scale, dim_name in [
            (scale_z, DimensionNames.SpatialZ),
            (scale_y, DimensionNames.SpatialY),
            (scale_x, DimensionNames.SpatialX),
        ]:
            if scale is not None and dim_name in dims_shape:
                dim_size = dims_shape[dim_name][1] - dims_shape[dim_name][0]
                coords[dim_name] = Reader._generate_coord_array(0, dim_size, scale)

        # Time
        # TODO: unpack "TimeSpan" elements
        # I can find a single "TimeSpan" in our data but unsure how multi-scene handles

        # Create physical pixel sizes
        px_sizes = types.PhysicalPixelSizes(scale_z, scale_y, scale_x)

        return coords, px_sizes

    def _read_delayed(self) -> xr.DataArray:
        """
        The delayed data array constructor for the image.

        Returns
        -------
        data: xr.DataArray
            The fully constructed delayed DataArray.

            It is additionally recommended to closely monitor how dask array chunks are
            managed.

        Notes
        -----
        Requirements for the returned xr.DataArray:
        * Must have the `dims` populated.
        * If a channel dimension is present, please populate the channel dimensions
        coordinate array the respective channel coordinate values.
        """

        def dim_indices(
            total_bounding_box: Dict[str, Tuple[int, int]], dim: str
        ) -> np.ndarray:
            """
            Example
            -------
            >>> dim_indices({
            ...   'X': (0, 100),
            ...   'Y': (0, 100)
            ... }, 'X') = np.array([0, 1, 2, ..., 99])
            """
            # TODO
            return (
                np.arange(total_bounding_box[dim][0], total_bounding_box[dim][1])
                * 4.57436261
            )

        def get_ordered_dims(
            total_bounding_box: Dict[str, Tuple[int, int]]
        ) -> List[str]:
            """
            Get the dimentions from the file in the default dimension order (TCZYX).

            Example
            -------
            >>> get_ordered_dims({
            ...   'X': (0, 100),
            ...   'Y': (0, 100),
            ...   'R': (0, 2)
            ... }) = ['R', 'Y', 'X']
            """
            available_dims = set(total_bounding_box.keys())
            available_default_dims = [
                d for d in DEFAULT_DIMENSION_ORDER_LIST if d in available_dims
            ]
            nondefault_dims = [
                d for d in available_dims if d not in DEFAULT_DIMENSION_ORDER_LIST
            ]
            ordered_dims = nondefault_dims + available_default_dims
            # TODO rewrite
            ordered_dims = [
                d
                for d in ordered_dims
                if total_bounding_box[d][1] - total_bounding_box[d][0] > 1
            ]
            # pylibCZIrw reads YX slices. Confirm that Y and X are the last two
            # dimensions.
            assert ordered_dims[-2:] == [
                DimensionNames.SpatialY,
                DimensionNames.SpatialX,
            ]
            return ordered_dims

        total_bounding_box = None
        pixel_types = None
        scenes_bounding_rectangle = None
        with open_czi_typed(self._path) as file:
            total_bounding_box = file.total_bounding_box
            pixel_types = file.pixel_types
            scenes_bounding_rectangle = file.scenes_bounding_rectangle

        ordered_dims = get_ordered_dims(total_bounding_box)
        # E.g., non_yx_dims = ['T', 'C', 'Z']
        non_yx_dims = ordered_dims[:-2]

        # coords = { d: dim_indices(total_bounding_box, d) for d in non_yx_dims}
        # TODO cleanup
        coords, _ = self._get_coords_and_physical_px_sizes(
            self.metadata, self._current_scene_index, total_bounding_box
        )
        assert (
            self._current_resolution_level in scenes_bounding_rectangle
        ), f"Expected {self._current_resolution_level} in {scenes_bounding_rectangle}."
        rect = scenes_bounding_rectangle[self._current_scene_index]
        coords.update(
            {
                DimensionNames.SpatialY: np.arange(rect.y, rect.y + rect.h)
                * 4.57436261,
                DimensionNames.SpatialX: np.arange(rect.x, rect.x + rect.w)
                * 4.57436261,
            }
        )

        # E.g., shape = (30, 2, 20, 100, 100)
        shape = tuple(len(coords[d]) for d in ordered_dims)
        print("shape", shape)
        # E.g., shape_without_yx = (30, 2, 20)
        shape_without_yx = shape[:-2]
        print("shape_without_yx", shape_without_yx)

        def array_builder(indices: tuple[int]) -> int:
            """
            Example
            -------
            >>> file: czi.CziReader
            >>> non_yx_dims = ['T', 'C', 'Z']
            >>> indices = [0, 1, 2]
            >>> array_builder(0, 1, 2) = file.read(
            ...   scene=0,
            ...   plane={'T': 0, 'C': 1, 'Z': 2}
            ... )
            """
            assert len(indices) >= len(
                non_yx_dims
            ), f"Expected {len(indices)} >= {len(non_yx_dims)}."
            plane = {d: indices[i] for i, d in enumerate(non_yx_dims)}
            with open_czi_typed(self._path) as file:
                result = file.read(scene=self._current_scene_index, plane=plane)
                # result.shape is (Y, X, 1) or (Y, X, 3) depending on whether it's RGB
                # or grayscale.
                return np.squeeze(result)
                if pixel_type_is_array(pixel_types[0]):
                    raise NotImplementedError(
                        f"RGB not supported. Pixel type: {pixel_types[0]}"
                    )
                    return result
                else:
                    return np.squeeze(result, axis=-1)

        # The Y and X shape of lazy_arrrays are both 1 because we are making each YX
        # slice a single chunk.
        # E.g., lazy_arrays.shape = (30, 2, 20, 1, 1)
        lazy_arrays: np.ndarray = np.ndarray(shape_without_yx + (1, 1), dtype=object)
        for np_index, _ in np.ndenumerate(lazy_arrays):
            lazy_arrays[np_index] = da.from_delayed(
                delayed(array_builder)(np_index),
                shape[-2:],  # TODO add (3,) if it's RGB
                dtype=PIXEL_DICT[pixel_types[0].lower()],
            )
        merged = da.block(lazy_arrays.tolist())
        print("merged.shape", merged.shape)

        return xr.DataArray(
            data=merged,
            dims=ordered_dims,
            coords=coords,
        )

    def _read_immediate(self) -> xr.DataArray:
        """
        The immediate data array constructor for the image.

        Returns
        -------
        data: xr.DataArray
            The fully read data array.

        Notes
        -----
        Requirements for the returned xr.DataArray:
        * Must have the `dims` populated.
        * If a channel dimension is present, please populate the channel dimensions
        coordinate array the respective channel coordinate values.
        """
        return self._read_delayed().compute()

    def _get_stitched_dask_mosaic(self) -> xr.DataArray:
        """
        Stitch all mosaic tiles back together and return as a single xr.DataArray with
        a delayed dask array for backing data.

        Returns
        -------
        mosaic: xr.DataArray
            The fully stitched together image. Contains all the dimensions of the image
            with the YX expanded to the full mosaic.

        Raises
        ------
        NotImplementedError
            Reader or format doesn't support reconstructing mosaic tiles.

        Notes
        -----
        Implementers can determine how to chunk the array.
        Most common is to chunk by tile.
        """
        raise NotImplementedError(
            "This reader does not support reconstructing mosaic images."
        )

    def _get_stitched_mosaic(self) -> xr.DataArray:
        """
        Stitch all mosaic tiles back together and return as a single xr.DataArray with
        an in-memory numpy array for backing data.

        Returns
        -------
        mosaic: np.ndarray
            The fully stitched together image. Contains all the dimensions of the image
            with the YX expanded to the full mosaic.

        Raises
        ------
        NotImplementedError
            Reader or format doesn't support reconstructing mosaic tiles.
        """
        raise NotImplementedError(
            "This reader does not support reconstructing mosaic images."
        )

    @property
    def xarray_dask_data(self) -> xr.DataArray:
        """
        Returns
        -------
        xarray_dask_data: xr.DataArray
            The delayed image and metadata as an annotated data array.
        """
        if self._xarray_dask_data is None:
            self._xarray_dask_data = self._read_delayed()

        return self._xarray_dask_data

    @property
    def xarray_data(self) -> xr.DataArray:
        """
        Returns
        -------
        xarray_data: xr.DataArray
            The fully read image and metadata as an annotated data array.
        """
        if self._xarray_data is None:
            self._xarray_data = self._read_immediate()

            # Remake the delayed xarray dataarray object using a rechunked dask array
            # from the just retrieved in-memory xarray dataarray
            self._xarray_dask_data = xr.DataArray(
                self._xarray_data.data,
                dims=self._xarray_data.dims,
                coords=self._xarray_data.coords,
                attrs=self._xarray_data.attrs,
            )

        return self._xarray_data

    @property
    def dask_data(self) -> da.Array:
        """
        Returns
        -------
        dask_data: da.Array
            The image as a dask array with the native dimension ordering.
        """
        return self.xarray_dask_data.data

    @property
    def data(self) -> np.ndarray:
        """
        Returns
        -------
        data: np.ndarray
            The image as a numpy array with native dimension ordering.
        """
        return self.xarray_data.data

    @property
    def mosaic_xarray_dask_data(self) -> xr.DataArray:
        """
        TODO Update comments on all the mosaic_ methods or just raise errors
        Returns
        -------
        xarray_dask_data: xr.DataArray
            The delayed mosaic image and metadata as an annotated data array.

        Raises
        ------
        InvalidDimensionOrderingError
            No MosaicTile dimension available to reader.

        Notes
        -----
        Each reader can implement mosaic tile stitching differently but it is common
        that each tile is a dask array chunk.
        """
        return self.xarray_dask_data

    @property
    def mosaic_xarray_data(self) -> xr.DataArray:
        """
        Returns
        -------
        xarray_dask_data: xr.DataArray
            The in-memory mosaic image and metadata as an annotated data array.

        Raises
        ------
        InvalidDimensionOrderingError
            No MosaicTile dimension available to reader.

        Notes
        -----
        Very large images should use `mosaic_xarray_dask_data` to avoid seg-faults.
        """
        return self.xarray_data

    @property
    def mosaic_dask_data(self) -> da.Array:
        """
        Returns
        -------
        dask_data: da.Array
            The stitched together mosaic image as a dask array.

        Raises
        ------
        InvalidDimensionOrderingError
            No MosaicTile dimension available to reader.

        Notes
        -----
        Each reader can implement mosaic tile stitching differently but it is common
        that each tile is a dask array chunk.
        """
        return self.xarray_dask_data.data

    @property
    def mosaic_data(self) -> np.ndarray:
        """
        Returns
        -------
        data: np.ndarray
            The stitched together mosaic image as a numpy array.

        Raises
        ------
        InvalidDimensionOrderingError
            No MosaicTile dimension available to reader.

        Notes
        -----
        Very large images should use `mosaic_dask_data` to avoid seg-faults.
        """
        return self.mosaic_xarray_data.data

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
        """
        if self._metadata is None:
            with open_czi_typed(self._path) as file:
                """
                Caution: The xml.etree.ElementTree module is not
                secure against maliciously constructed data. If you need
                to parse untrusted or unauthenticated data see XML vulnerabilities.
                TODO highlight this better? aicspylibczi does the same thing though
                """
                self._metadata = ElementTree.fromstring(file.raw_metadata)
        return self._metadata

    @property
    def ome_metadata(self) -> OME:
        """
        Returns
        -------
        metadata: OME
            The original metadata transformed into the OME specfication.
            This likely isn't a complete transformation but is guarenteed to
            be a valid transformation.

        Raises
        ------
        NotImplementedError
            No metadata transformer available.
        """
        raise NotImplementedError()

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

        def physical_pixel_size(
            metadata: Metadata, dimension: str, allow_none: bool = False
        ) -> float | None:
            scales = metadata.findall(
                f"./Metadata/Scaling/Items/Distance[@Id='{dimension}']"
            )

            if len(scales) != 1:
                if allow_none and len(scales) == 0:
                    return None
                raise UnsupportedMetadataError(
                    f"Expected 1 distance scale for dimension '{dimension}' but found "
                    f"{len(scales)}."
                )

            unparsed_scale = scales[0].find("./Value")
            if unparsed_scale is None or unparsed_scale.text is None:
                raise UnsupportedMetadataError(
                    f"Could not find any distance scale for dimension '{dimension}'."
                )
            scale_m = float(unparsed_scale.text)
            # The values are stored in units of meters always in .czi. Convert to
            # microns.
            return scale_m / 1e-6

        return PhysicalPixelSizes(
            Z=physical_pixel_size(self.metadata, "Z", allow_none=True),
            Y=physical_pixel_size(self.metadata, "Y"),
            X=physical_pixel_size(self.metadata, "X"),
        )

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


def open_czi_typed(
    filepath: str,
    file_input_type: czi.ReaderFileInputTypes = czi.ReaderFileInputTypes.Standard,
    cache_options: czi.CacheOptions | None = None,
) -> ContextManager[czi.CziReader]:
    """
    Wrapper around czi.open_czi to provide type hinting that clarifies the result
    is a czi.CziReader
    """
    return czi.open_czi(filepath, file_input_type, cache_options)


class UnsupportedMetadataError(Exception):
    """
    The reader encountered metadata it doesn't know how to handle.
    """
