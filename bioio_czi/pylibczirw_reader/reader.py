#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import logging
from typing import Any, Callable, ContextManager, Dict, List, Optional, Tuple, Union
from xml.etree import ElementTree as ET

import dask.array as da
import numpy as np
import xarray as xr
from bioio_base import constants, exceptions, types
from bioio_base.dimensions import (
    DEFAULT_DIMENSION_ORDER_LIST,
    DimensionNames,
    Dimensions,
)
from bioio_base.reader import Reader as BaseReader
from bioio_base.transforms import reshape_data
from bioio_base.types import PhysicalPixelSizes
from dask import delayed
from fsspec.spec import AbstractFileSystem
from pylibCZIrw import czi

from .. import metadata
from ..channels import get_channel_names, size
from ..metadata import UnsupportedMetadataError
from ..pixel_sizes import get_physical_pixel_sizes

log = logging.getLogger(__name__)

PIXEL_DICT = {
    "gray8": np.uint8,
    "gray16": np.uint16,
    "gray32": np.uint32,  # Not supported by underlying pylibCZIrw
    "gray32float": np.float32,
    "bgr24": np.uint8,
    "bgr48": np.uint16,
    "bgr96float": np.float32,  # Supported by pylibCZIrw but not tested in this plugin
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

    NAME = "bioio-czi-pylibczirw"

    _xarray_dask_data: Optional["xr.DataArray"] = None
    _xarray_data: Optional["xr.DataArray"] = None
    _dims: Optional[Dimensions] = None
    _metadata: Optional[ET.Element] = None
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
            Boolean value indicating if the file is supported by the reader,
            or raises an exception if it is not.
        """
        try:
            with open(path):
                return True
        except RuntimeError as e:
            raise exceptions.UnsupportedFileFormatError(
                "bioio-czi[pylibczirw mode]",
                path,
                str(e),
            )

    def __init__(self, image: types.PathLike, fs_kwargs: Dict[str, Any] = {}) -> None:
        path = str(image)
        try:
            with open(path) as file:
                self._fs = None  # Unused but required by tests
                self._path = path
                self._total_bounding_box = file.total_bounding_box_no_pyramid
                self._pixel_types = file.pixel_types
                self._scenes_bounding_rectangle = (
                    file.scenes_bounding_rectangle_no_pyramid
                )
                self._czi_scene_indices = sorted(self._scenes_bounding_rectangle.keys())
        except RuntimeError:
            raise exceptions.UnsupportedFileFormatError(self.__class__.__name__, path)

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

        def scene_name(metadata: ET.Element, czi_scene_index: int) -> str:
            scene_info = metadata.findall(
                "./Metadata/Information/Image/Dimensions/"
                f"S/Scenes/Scene[@Index='{czi_scene_index}']"
            )
            if len(scene_info) != 1:
                raise UnsupportedMetadataError(
                    f"Expected 1 scene for index '{czi_scene_index}' "
                    f"but found {len(scene_info)}."
                )
            scene_name = scene_info[0].get("Name")
            if scene_name is None:
                scene_name = str(czi_scene_index)

            shape_info = scene_info[0].find("Shape")
            if shape_info is not None:
                shape_name = shape_info.get("Name")
                if shape_name is not None:
                    return f"{scene_name}-{shape_name}"

            return scene_name

        if self._scenes is None:
            # Case 1: We have scenes with bounding rectangles (normal CZI)
            if hasattr(self, "_czi_scene_indices") and len(self._czi_scene_indices) > 0:
                self._scenes = tuple(
                    scene_name(self.metadata, czi_index)
                    for czi_index in self._czi_scene_indices
                )

            # Case 2: No scene info, fall back to a default scene
            if not self._scenes:
                self._scenes = (metadata.generate_ome_image_id(0),)

        return self._scenes

    def _get_coords(
        self, xml: ET.Element, scene_index: int, dims_shape: Dict[str, Any]
    ) -> Dict[str, Union[list, np.ndarray]]:
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

    @property
    def czi_scene_index(self) -> int:
        """
        Returns the plate-level CZI scene index for the current scene.

        For split CZI files (one plate position per file), this differs from
        current_scene_index (which is always 0) because the embedded XML metadata
        still uses the original plate-wide scene indices.
        """
        return self._get_czi_scene_index()

    def _get_czi_scene_index(self, scene_index: Optional[int] = None) -> int:
        """
        Map a BioIO scene index (0..N-1) to the underlying CZI scene index.

        If no explicit scenes in the CZI (len(_scenes_bounding_rectangle) == 0),
        we just return 0 and rely on pylibczirw's defaults (scene=None handlers).
        """
        if len(self._scenes_bounding_rectangle) == 0:
            return 0

        if scene_index is None:
            scene_index = self._current_scene_index

        if scene_index < 0 or scene_index >= len(self._czi_scene_indices):
            raise IndexError(
                f"BioIO scene index {scene_index} is out of range for "
                f"{len(self._czi_scene_indices)} scenes."
            )

        return self._czi_scene_indices[scene_index]

    def _array_builder(self, index_dims: list[str]) -> Callable[[tuple[int]], int]:
        """
        Internal helper method to get one chunk of the image data.

        Parameters
        ----------
        index_dims: list[str]
            The names of the dimensions that will be used to select a chunk.

        Returns
        -------
        array_builder: Callable[[tuple[int]], int]
            Function of one parameter indices: tuple[int] that defines the chunk.
            indices must be the same length as index_dims, and in the same order.

        Example
        -------
        >>> self._array_builder(['T', 'C', 'Z'])((0, 1, 2)) = file.read(
        ...   scene=0,
        ...   plane={'T': 0, 'C': 1, 'Z': 2}
        ... )
        """

        # Freeze the scene / ROI for lazy builder invocation.
        if len(self._scenes_bounding_rectangle) == 0:
            # Some files have no scenes but can still be read if scene is not
            # specified.
            current_scene: int | None = None
            current_roi = None
        else:
            czi_scene_index = self._get_czi_scene_index()
            current_scene = czi_scene_index
            current_roi = self._scenes_bounding_rectangle[czi_scene_index]

        def array_builder(indices: tuple[int]) -> int:
            assert len(indices) >= len(
                index_dims
            ), f"Expected {len(indices)} >= {len(index_dims)}."
            # E.g., plane = {'T': 0, 'C': 1, 'Z': 2}
            plane = {d: indices[i] for i, d in enumerate(index_dims)}
            # The purpose of the next 2 lines is complicated.
            # ROI stands for Region Of Interest.
            #
            # In pylibczi's read method, the default ROI is the bounding
            # rectangle of the scene **across all zoom levels**. We are going to
            # read just the highest resolution level (zoom = 1), which is
            # smaller than the default ROI in some cases. For example,
            # scene 0 of the test file S=2_4x2_T=2=Z=3_CH=2.czi is 947x487 when
            # looking at only the highest resolution, but is 948x488 when
            # all zoom levels are considered. (I believe this is because at zoom
            # 0.5, the result is ceiling(947/2) x ceiling(487/2).)
            #
            # See also: file.scenes_bounding_rectangle vs.
            # file.scenes_bounding_rectangle_no_pyramid.
            #
            # Therefore, when calling read, we crop to just the ROI of the
            # highest resolution level.
            #
            # NOTE: self._current_scene_index is a BioIO scene index (0..N-1).
            # We must map it to the underlying CZI scene index before using it
            # with pylibczirw or _scenes_bounding_rectangle.
            with open(self._path) as file:
                result = file.read(scene=current_scene, plane=plane, roi=current_roi)
            # result.shape is (Y, X, 1) or (Y, X, 3) depending on whether it's RGB
            # or grayscale. We want to return (Y, X) or (Y, X, 3).
            return np.squeeze(result)

        return array_builder

    def _read_delayed(self) -> xr.DataArray:
        """
        The delayed data array constructor for the image.

        Returns
        -------
        data: xr.DataArray
            The fully constructed delayed DataArray.

            It is additionally recommended to closely monitor how dask array chunks are
            managed.
        """
        # 1. Combine the dimension bounds from total_bounding_box (all dimensions) and
        # scenes_bounding_rectangle (XY only) in order to compute the coordinate array
        # for each dimension. (Think of the coordinate array as the "ticks" on an axis.)
        dim_bounds = self._total_bounding_box
        if len(self._scenes_bounding_rectangle) > 0:
            czi_scene_index = self._get_czi_scene_index()
            assert czi_scene_index in self._scenes_bounding_rectangle, (
                f"Expected CZI scene index {czi_scene_index} (from BioIO index "
                f"{self._current_scene_index}) to be in "
                f"{self._scenes_bounding_rectangle}."
            )
            rect = self._scenes_bounding_rectangle[czi_scene_index]
            dim_bounds[DimensionNames.SpatialX] = (rect.x, rect.x + rect.w)
            dim_bounds[DimensionNames.SpatialY] = (rect.y, rect.y + rect.h)

        coords = self._get_coords(
            self.metadata,
            self._get_czi_scene_index(),
            dim_bounds,
        )

        # 2. Figure out which dimensions are available on this image, and put them in
        # TCZYX order as much as possible.
        ordered_dims = [
            d
            for d in DEFAULT_DIMENSION_ORDER_LIST
            if d in coords or size(self._total_bounding_box, d) > 1
        ]
        assert ordered_dims[-2:] == [DimensionNames.SpatialY, DimensionNames.SpatialX]
        # E.g., non_yx_dims = ['T', 'C', 'Z']
        non_yx_dims = ordered_dims[:-2]

        # 4. Determine the chunk sizes and number of chunks. Each chunk is a single
        # YX slice.
        # E.g., shape = (30, 2, 20, 100, 100)
        shape = tuple(
            len(coords[d]) if d in coords else size(self._total_bounding_box, d)
            for d in ordered_dims
        )
        # The Y and X shape of lazy_arrays are both 1 because we are making each YX
        # slice a single chunk.
        # E.g., lazy_arrays.shape = (30, 2, 20, 1, 1)
        shape_for_lazyarrays = shape[:-2] + (1, 1)

        chunk_shape = shape[-2:]
        if "Bgr" in self._pixel_types[0]:
            # If the image is BGR, each chunk has shape (X, Y, 3)...
            chunk_shape += (3,)
            ordered_dims.append(DimensionNames.Samples)
            # ...and we also need to reflect this in the shape of the lazy_arrays
            # All the Y, X *and* S points for a slice are in a single chunk
            shape_for_lazyarrays += (1,)

        # 5. Create delayed chunks
        lazy_arrays: np.ndarray = np.ndarray(shape_for_lazyarrays, dtype=object)
        for np_index, _ in np.ndenumerate(lazy_arrays):
            lazy_arrays[np_index] = da.from_delayed(
                delayed(self._array_builder(non_yx_dims))(np_index),
                chunk_shape,
                dtype=PIXEL_DICT[self._pixel_types[0].lower()],
            )

        # 6. Package chunks and metadata into a DataArray
        return xr.DataArray(
            data=da.block(lazy_arrays.tolist()),
            dims=ordered_dims,
            coords=coords,
            attrs={constants.METADATA_UNPROCESSED: self.metadata},
        )

    def _read_immediate(self) -> xr.DataArray:
        """
        The immediate data array constructor for the image.

        Returns
        -------
        data: xr.DataArray
            The fully read data array.
        """
        return self._read_delayed().compute()

    def _scene_dims_and_shape(self) -> Tuple[str, Tuple[int, ...]]:
        """
        Native dimension order and shape of the current scene, derived from the
        bounding boxes alone -- i.e. without building the per-plane dask graph
        that ``self.dims`` / ``self.shape`` would trigger via ``_read_delayed``.

        Mirrors the dimension/shape bookkeeping in ``_read_delayed`` (including
        the trailing ``Samples`` axis for BGR images); keep the two in sync.
        """
        dim_bounds = dict(self._total_bounding_box)
        if len(self._scenes_bounding_rectangle) > 0:
            rect = self._scenes_bounding_rectangle[self._get_czi_scene_index()]
            dim_bounds[DimensionNames.SpatialX] = (rect.x, rect.x + rect.w)
            dim_bounds[DimensionNames.SpatialY] = (rect.y, rect.y + rect.h)

        coords = self._get_coords(
            self.metadata, self._get_czi_scene_index(), dim_bounds
        )
        ordered_dims = [
            d
            for d in DEFAULT_DIMENSION_ORDER_LIST
            if d in coords or size(self._total_bounding_box, d) > 1
        ]
        assert ordered_dims[-2:] == [
            DimensionNames.SpatialY,
            DimensionNames.SpatialX,
        ]
        shape = tuple(
            len(coords[d]) if d in coords else size(self._total_bounding_box, d)
            for d in ordered_dims
        )
        if "Bgr" in self._pixel_types[0]:
            ordered_dims = ordered_dims + [DimensionNames.Samples]
            shape = shape + (3,)
        return "".join(ordered_dims), shape

    def _read_region(
        self,
        dimension_order_out: Optional[str] = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """
        Read a hyper-rectangular region of the current scene directly from the
        file, holding it open for the whole region. This is the backbone of
        :meth:`get_image_data` for sliced selections; it is not public API.

        Unlike ``get_image_dask_data(...).compute()`` -- which builds a
        per-plane dask graph and re-opens the file for every YX slice -- this
        opens the CZI once and streams only the requested planes, cropping each
        to the requested XY sub-region via a pylibCZIrw ROI. It never
        materializes the full image, so partial reads are both faster and
        memory-bounded.

        Parameters
        ----------
        dimension_order_out: Optional[str]
            Desired dimension order of the result. Default: the image's native
            order. Dimensions selected by a scalar ``int`` and absent from this
            order are dropped, matching ``get_image_data``.
        kwargs: Any
            Per-dimension selection. Each value is either an ``int`` (a single
            index) or a contiguous ``slice`` (``step`` must be ``1`` or
            ``None``). Unspecified dimensions are read in full. ``X``/``Y``
            selections become the ROI; all other dimensions are iterated
            plane-by-plane.

        Returns
        -------
        np.ndarray
            The region, in ``dimension_order_out``.
        """
        native_order, native_shape = self._scene_dims_and_shape()
        dim_sizes = dict(zip(native_order, native_shape))
        frame_axes = {
            DimensionNames.SpatialY,
            DimensionNames.SpatialX,
            DimensionNames.Samples,
        }

        # Resolve each native dimension to a (start, stop) read window.
        windows: Dict[str, Tuple[int, int]] = {}
        for d in native_order:
            if d not in kwargs:
                windows[d] = (0, dim_sizes[d])
            elif isinstance(kwargs[d], slice):
                if kwargs[d].step not in (None, 1):
                    raise ValueError(
                        "read_region only supports contiguous slices; got step "
                        f"{kwargs[d].step} for dimension {d!r}."
                    )
                start, stop, _ = kwargs[d].indices(dim_sizes[d])
                windows[d] = (start, stop)
            elif isinstance(kwargs[d], (int, np.integer)):
                idx = int(kwargs[d]) % dim_sizes[d]
                windows[d] = (idx, idx + 1)
            else:
                raise TypeError(
                    f"read_region selection for {d!r} must be int or slice, got "
                    f"{type(kwargs[d]).__name__}."
                )

        # XY origin: per-scene for scened files, total bounding box otherwise.
        if len(self._scenes_bounding_rectangle) == 0:
            scene: Optional[int] = None
            base_x = self._total_bounding_box[DimensionNames.SpatialX][0]
            base_y = self._total_bounding_box[DimensionNames.SpatialY][0]
        else:
            scene = self._get_czi_scene_index()
            rect = self._scenes_bounding_rectangle[scene]
            base_x, base_y = rect.x, rect.y

        x0, x1 = windows[DimensionNames.SpatialX]
        y0, y1 = windows[DimensionNames.SpatialY]
        roi = (base_x + x0, base_y + y0, x1 - x0, y1 - y0)

        out_shape = tuple(windows[d][1] - windows[d][0] for d in native_order)
        region = np.empty(out_shape, dtype=PIXEL_DICT[self._pixel_types[0].lower()])
        has_samples = DimensionNames.Samples in native_order

        loop_axes = [d for d in native_order if d not in frame_axes]
        ranges = [range(*windows[d]) for d in loop_axes]

        with open(self._path) as file:
            for combo in itertools.product(*ranges):
                plane = {d: combo[i] for i, d in enumerate(loop_axes)}
                raw = file.read(scene=scene, plane=plane, roi=roi)
                # raw is (Y, X, 1) grayscale or (Y, X, 3) BGR.
                result = raw if has_samples else raw[..., 0]
                out_idx: List[Any] = [slice(None)] * len(native_order)
                for i, d in enumerate(loop_axes):
                    out_idx[native_order.index(d)] = combo[i] - windows[d][0]
                region[tuple(out_idx)] = result

        if dimension_order_out is None or dimension_order_out == native_order:
            return region
        return reshape_data(
            data=region,
            given_dims=native_order,
            return_dims=dimension_order_out,
        )

    def get_image_data(
        self, dimension_order_out: Optional[str] = None, **kwargs: Any
    ) -> np.ndarray:
        """
        Read specific dimension image data as a numpy array.

        When the selection is a hyper-rectangle -- every keyword is an ``int``
        or a contiguous ``slice`` and at least one is a ``slice`` -- this reads
        only the requested bytes directly from the file (holding it open for the
        whole region) instead of materializing the whole image first. All other
        selections (lists, ranges, strided slices, or no keywords) defer to the
        base implementation. See the base ``Reader.get_image_data`` for
        parameter details.
        """
        routable = (
            bool(kwargs)
            and any(isinstance(v, slice) for v in kwargs.values())
            and all(
                isinstance(v, (int, np.integer))
                or (isinstance(v, slice) and v.step in (None, 1))
                for v in kwargs.values()
            )
        )
        if routable:
            return self._read_region(dimension_order_out, **kwargs)
        return super().get_image_data(dimension_order_out, **kwargs)

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
    def metadata(self) -> ET.Element:
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
                self._metadata = ET.fromstring(file.raw_metadata)
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

    @property
    def total_time_duration(self) -> None:
        """
        Cannot read time duration accurately without subblock metadata.
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
