#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import logging
from typing import (
    Any,
    Callable,
    ContextManager,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)
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
from bioio_base.types import PhysicalPixelSizes
from dask import delayed
from fsspec.spec import AbstractFileSystem
from pylibCZIrw import czi

from .. import metadata, remote
from ..channels import get_channel_names, size
from ..metadata import UnsupportedMetadataError
from ..pixel_sizes import get_physical_pixel_sizes

log = logging.getLogger(__name__)

READER_NAME = "bioio-czi[pylibczirw mode]"

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
        Path to image file to construct Reader for. May be a local path, an http(s)
        URL, or an object-store URI such as "s3://bucket/key". See the Notes section
        for how remote images are read.
    fs_kwargs: Dict[str, Any]
        Any specific keyword arguments to pass down to the fsspec created filesystem.
        Only used to presign object-store URIs; local paths and http(s) URLs go
        straight to libCZI.
        Default: {}

    Notes
    -----
    Remote images are read by libCZI's curl-based stream, which issues range
    requests for just the sub-blocks needed rather than downloading the whole file.
    The server must support range requests. Protocols other than http(s) are
    presigned into an https URL by their fsspec filesystem, so credentials are
    resolved by fsspec in the usual way.
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
            with open(path, fs_kwargs=kwargs.get("fs_kwargs")):
                return True
        except exceptions.UnsupportedFileFormatError:
            # Already explains itself, e.g. a protocol that cannot be presigned.
            raise
        except (RuntimeError, OSError) as e:
            # libCZI reports an unreadable file as a RuntimeError. Reading over the
            # network adds connection and HTTP failures on top, which arrive as
            # OSError, and mean "could not fetch" rather than "not a CZI" -- but
            # either way this reader cannot open the image.
            raise exceptions.UnsupportedFileFormatError(
                READER_NAME,
                path,
                str(e),
            )

    def __init__(self, image: types.PathLike, fs_kwargs: Dict[str, Any] = {}) -> None:
        path = str(image)
        self._fs_kwargs = fs_kwargs
        try:
            with open(path, fs_kwargs=fs_kwargs) as file:
                self._fs = None  # Unused but required by tests
                self._path = path
                self._total_bounding_box = file.total_bounding_box_no_pyramid
                self._pixel_types = file.pixel_types
                self._scenes_bounding_rectangle = (
                    file.scenes_bounding_rectangle_no_pyramid
                )
                self._czi_scene_indices = sorted(self._scenes_bounding_rectangle.keys())
        except (RuntimeError, OSError) as e:
            # See _is_supported_image for why OSError is caught alongside RuntimeError.
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__, path, str(e)
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
        current_scene, current_roi = self._current_scene_roi()

        def array_builder(indices: tuple[int]) -> int:
            assert len(indices) >= len(
                index_dims
            ), f"Expected {len(indices)} >= {len(index_dims)}."
            # E.g., plane = {'T': 0, 'C': 1, 'Z': 2}
            plane = {d: indices[i] for i, d in enumerate(index_dims)}
            with open(self._path, fs_kwargs=self._fs_kwargs) as file:
                result = file.read(scene=current_scene, plane=plane, roi=current_roi)
            # result.shape is (Y, X, 1) or (Y, X, 3) depending on whether it's RGB
            # or grayscale. We want to return (Y, X) or (Y, X, 3).
            return np.squeeze(result)

        return array_builder

    def _current_scene_roi(self) -> Tuple[Optional[int], Any]:
        """
        Resolve the ``(czi_scene_index, highest-resolution ROI)`` for the current
        scene, for use with ``file.read(scene=, roi=)``.

        ROI stands for Region Of Interest. In pylibczi's read method, the default
        ROI is the bounding rectangle of the scene **across all zoom levels**. We
        read just the highest resolution level (zoom = 1), which is smaller than
        the default ROI in some cases. For example, scene 0 of the test file
        S=2_4x2_T=2=Z=3_CH=2.czi is 947x487 at the highest resolution, but 948x488
        when all zoom levels are considered. (At zoom 0.5 the result is
        ceiling(947/2) x ceiling(487/2).) See also file.scenes_bounding_rectangle
        vs. file.scenes_bounding_rectangle_no_pyramid.

        NOTE: self._current_scene_index is a BioIO scene index (0..N-1); it is
        mapped to the underlying CZI scene index before use with pylibczirw or
        _scenes_bounding_rectangle.
        """
        # Some files have no scenes but can still be read if scene is not specified.
        if len(self._scenes_bounding_rectangle) == 0:
            return None, None
        czi_scene_index = self._get_czi_scene_index()
        return czi_scene_index, self._scenes_bounding_rectangle[czi_scene_index]

    @property
    def dims(self) -> Dimensions:
        """
        Dimension names and sizes of the current scene.

        We override here to fetch from CZI metadata directly instead of
        deriving it from ``xarray_dask_data`` (the base implementation).

        Returns
        -------
        dims: Dimensions
            Object with the paired dimension names and their sizes.
        """
        if self._dims is None:
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
            self._dims = Dimensions(dims="".join(ordered_dims), shape=shape)
        return self._dims

    @property
    def shape(self) -> Tuple[int, ...]:
        """
        Shape of the current scene.

        We override here to fetch from CZI metadata directly instead of
        deriving it from ``xarray_dask_data`` (the base implementation).

        Returns
        -------
        shape: Tuple[int, ...]
            Tuple of the image array's dimensions.
        """
        return self.dims.shape

    @staticmethod
    def _spatial_window(
        spec: Union[int, slice, list], size: int
    ) -> Tuple[int, int, Any]:
        """
        pylibCZIrw reads pixels off disk as a single contiguous rectangle (an
        ROI), but a caller can select this axis with an int, a slice (possibly
        strided), or a list of indices -- none of which an ROI expresses
        directly. So we split the selection into two pieces:

        - ``(start, extent)``: the smallest contiguous range covering everything
          the caller asked for. This is the ROI we actually read off disk.
        - ``residual``: the index then applied to that read window, in memory, to
          recover the exact selection -- drop the axis (int), re-apply the stride
          (strided slice), or pick out the listed positions (list).

        Reading the bounding range and re-indexing keeps disk reads small while
        still honoring selections an ROI alone cannot express.

        Returns ``(start, extent, residual)`` for the given ``spec`` and axis
        ``size``.
        """
        if isinstance(spec, (int, np.integer)):
            k = int(spec) % size
            return k, 1, 0
        if isinstance(spec, slice):
            start, stop, step = spec.indices(size)
            extent = max(stop - start, 0)
            if step == 1:
                return start, extent, slice(None)
            return start, extent, slice(0, extent, step)
        if isinstance(spec, list):
            idxs = [j % size for j in spec]
            lo, hi = min(idxs), max(idxs)
            return lo, hi - lo + 1, [j - lo for j in idxs]
        raise TypeError(
            f"Spatial selection must be int, slice, or list, got "
            f"{type(spec).__name__}."
        )

    def _read_indexed(self, given_dims: str, dim_specs: list) -> np.ndarray:
        """
        Return the native-order array with ``dim_specs`` applied.

        This lets ``get_image_data`` read only the requested sub-region. It
        reads each plane one at a time and translates the Y/X selection into a
        single ROI, so only the requested area is read off disk.

        Parameters
        ----------
        given_dims: str
            The native dimension ordering of the image (``self.dims.order``).
        dim_specs: list
            One getitem operation per dimension in ``given_dims``, as produced by
            ``transforms.compute_dim_specs``.

        Returns
        -------
        data: np.ndarray
            The indexed image data in native (reduced) dimension order.
        """
        native_shape = self.shape
        y_index = given_dims.index(DimensionNames.SpatialY)
        cullable_dims = given_dims[:y_index]  # e.g. "TCZ"
        has_samples = DimensionNames.Samples in given_dims

        # Fetch YX rectangle
        y_start, y_extent, y_residual = self._spatial_window(
            dim_specs[y_index], native_shape[y_index]
        )
        x_start, x_extent, x_residual = self._spatial_window(
            dim_specs[y_index + 1], native_shape[y_index + 1]
        )
        window_specs: List[Any] = [y_residual, x_residual]
        if has_samples:
            window_specs.append(dim_specs[y_index + 2])  # Samples spec
        window_specs_t = tuple(window_specs)

        # XY origin: per-scene for scened files, total bounding box otherwise.
        if len(self._scenes_bounding_rectangle) == 0:
            scene: Optional[int] = None
            base_x = self._total_bounding_box[DimensionNames.SpatialX][0]
            base_y = self._total_bounding_box[DimensionNames.SpatialY][0]
        else:
            scene = self._get_czi_scene_index()
            rect = self._scenes_bounding_rectangle[scene]
            base_x, base_y = rect.x, rect.y
        roi = (base_x + x_start, base_y + y_start, x_extent, y_extent)

        # Resolve each cullable dim to (out_pos | None, plane_index) tuples.
        # out_pos is None for fixed (integer) dims whose axis is dropped.
        enumerated: List[list] = []
        for i, _dim in enumerate(cullable_dims):
            spec = dim_specs[i]
            size_i = native_shape[i]
            if isinstance(spec, slice):
                enumerated.append(list(enumerate(range(*spec.indices(size_i)))))
            elif isinstance(spec, list):
                enumerated.append(list(enumerate([j % size_i for j in spec])))
            else:  # int -> fixed, axis dropped
                enumerated.append([(None, int(spec) % size_i)])

        kept_lengths = [
            len(e)
            for e, spec in zip(enumerated, dim_specs[:y_index])
            if not isinstance(spec, (int, np.integer))
        ]

        out: Optional[np.ndarray] = None
        with open(self._path, fs_kwargs=self._fs_kwargs) as file:
            for combo in itertools.product(*enumerated):
                out_pos = tuple(pos for pos, _idx in combo if pos is not None)
                plane = {d: idx for (d, (_pos, idx)) in zip(cullable_dims, combo)}
                raw = file.read(scene=scene, plane=plane, roi=roi)
                # raw is (Y, X, 1) grayscale or (Y, X, 3) BGR. Drop the trailing
                # samples axis for grayscale.
                result = raw if has_samples else raw[..., 0]
                cropped = result[window_specs_t]
                if out is None:
                    out = np.empty(
                        tuple(kept_lengths) + cropped.shape, dtype=cropped.dtype
                    )
                out[out_pos] = cropped

        if out is None:
            # A dim selected nothing (e.g. C=slice(0, 0)) so the read
            # loop never ran; build a full-dimensionality empty result.
            spatial_template: Tuple[int, ...] = (y_extent, x_extent)
            if has_samples:
                spatial_template += (
                    native_shape[given_dims.index(DimensionNames.Samples)],
                )
            spatial_shape = np.empty(spatial_template)[window_specs_t].shape
            out = np.empty(
                tuple(kept_lengths) + spatial_shape,
                dtype=PIXEL_DICT[self._pixel_types[0].lower()],
            )
        return out

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
            with open(self._path, fs_kwargs=self._fs_kwargs) as file:
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


def open(
    filepath: str, fs_kwargs: Optional[Dict[str, Any]] = None
) -> ContextManager[czi.CziReader]:
    """
    Open a CZI wherever it lives, local or remote.

    Also wraps czi.open_czi to provide type hinting that clarifies the result is a
    czi.CziReader.

    Parameters
    ----------
    filepath: str
        A local path, an http(s) URL, or an object-store URI such as
        "s3://bucket/key".
    fs_kwargs: Optional[Dict[str, Any]]
        Keyword arguments for the fsspec filesystem used to presign object-store
        URIs. Ignored for local paths and http(s) URLs.
        Default: None

    Notes
    -----
    Remote images are read through libCZI's curl-based stream, which fetches only
    the byte ranges it needs rather than downloading the whole file. Protocols
    other than http(s) are presigned into an https URL first, so credentials are
    resolved by fsspec and never handed to libCZI.
    """
    if remote.is_remote(filepath):
        url = remote.resolve_url(filepath, reader_name=READER_NAME, fs_kwargs=fs_kwargs)
        return czi.open_czi(url, czi.ReaderFileInputTypes.Curl)
    return czi.open_czi(filepath)
