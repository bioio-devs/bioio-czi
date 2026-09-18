import functools
import itertools
import logging
import warnings
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from copy import copy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, Hashable, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr
from _aicspylibczi import BBox
from aicspylibczi import CziFile
from bioio_base import constants, exceptions
from bioio_base import io as io_utils
from bioio_base import types
from bioio_base.dimensions import (
    DEFAULT_CHUNK_DIMS,
    REQUIRED_CHUNK_DIMS,
    DimensionNames,
    Dimensions,
)
from bioio_base.reader import Reader as BaseReader
from bioio_base.standard_metadata import StandardMetadata
from dask import delayed
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem
from ome_types.model import OME

from . import metadata as metadata_utils
from . import standard_metadata as standard_metadata_utils
from .bounding_box import size
from .channels import get_channel_names
from .pixel_sizes import get_physical_pixel_sizes
from .subblock_metadata import acquisition_times, time_between_subblocks

###############################################################################

log = logging.getLogger(__name__)

###############################################################################

# Sentinel so we can tell whether the deprecated use_aicspylibczi kwarg was passed.
_USE_AICSPYLIBCZI_UNSET = object()

CZI_SAMPLES_DIM_CHAR = "A"
CZI_BLOCK_DIM_CHAR = "B"
CZI_SCENE_DIM_CHAR = "S"

# Maps BioIO dimension chars back to CZI-native read_image chars. Only Samples
# differs (BioIO "S" <- CZI "A"); all other non-spatial dims (T/C/Z/M/...) are
# identical in both schemes.
_BIOIO_TO_CZI_DIM = {DimensionNames.Samples: CZI_SAMPLES_DIM_CHAR}


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


@functools.lru_cache(maxsize=16)
def _remote_czi(url: str, stream_options: Tuple[Tuple[str, Any], ...]) -> CziFile:
    return CziFile(url, stream_options=dict(stream_options))


@contextmanager
def open_czi(
    fs: AbstractFileSystem, path: str, stream_options: Optional[Dict[str, Any]] = None
) -> Generator[CziFile, None, None]:
    """
    Yield a CziFile for ``path``. Local files are opened per call so no handle is
    held between calls; http(s) URLs reuse one cached handle per URL and options, as
    opening a remote CZI refetches its header, metadata and subblock directory.
    """
    if isinstance(fs, LocalFileSystem):
        with fs.open(path) as open_resource:
            yield CziFile(open_resource.f)
    else:
        yield _remote_czi(path, tuple(sorted((stream_options or {}).items())))


class Reader(BaseReader):
    """
    A BioIO Reader plugin for volumetric Zeiss CZI images, backed by aicspylibczi.

    Notes
    -----
    To use this reader, install with: `pip install aicspylibczi>=4.0.1`.
    """

    NAME = "bioio-czi-aicspylibczi"

    _xarray_dask_data: Optional["xr.DataArray"] = None
    _xarray_data: Optional["xr.DataArray"] = None
    _mosaic_xarray_dask_data: Optional["xr.DataArray"] = None
    _mosaic_xarray_data: Optional["xr.DataArray"] = None
    _dims: Optional[Dimensions] = None
    _dtype: Optional[np.dtype] = None
    _metadata: Optional[Any] = None
    _scenes: Optional[Tuple[str, ...]] = None
    _current_scene_index: int = 0
    # Do not provide default value because
    # they may not need to be used by your reader (i.e. input param is an array)
    _fs: "AbstractFileSystem"
    _path: str

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        if not isinstance(fs, LocalFileSystem) and not CziFile.is_remote(path):
            raise exceptions.UnsupportedFileFormatError(
                "bioio-czi[aicspylibczi mode]",
                path,
                "Only local paths and http(s) URLs are supported.",
            )
        try:
            with open_czi(fs, path, kwargs.get("stream_options")):
                return True
        except RuntimeError as e:
            raise exceptions.UnsupportedFileFormatError(
                "bioio-czi[aicspylibczi mode]",
                path,
                str(e),
            )

    def __init__(
        self,
        image: types.PathLike,
        chunk_dims: Union[str, List[str]] = DEFAULT_CHUNK_DIMS,
        include_subblock_metadata: bool = False,
        fs_kwargs: Dict[str, Any] = {},
        stream_options: Optional[Dict[str, Any]] = None,
        use_aicspylibczi: Any = _USE_AICSPYLIBCZI_UNSET,
    ):
        """
        Parameters
        ----------
        image: types.PathLike
            Path to image file to construct Reader for, or an http(s) URL.
        chunk_dims: Union[str, List[str]]
            Which dimensions to create chunks for.
            Default: DEFAULT_CHUNK_DIMS
            Note: DimensionNames.SpatialY, DimensionNames.SpatialX, and
            DimensionNames.Samples, will always be added to the list if not present
            during dask array construction.
        include_subblock_metadata: bool
            Whether to append metadata from the subblocks to the rest of the embeded
            metadata.
        fs_kwargs: Dict[str, Any]
            Any specific keyword arguments to pass to the fsspec-created filesystem.
            For http(s) URLs this only affects checking that the file exists; the
            read itself is configured with stream_options.
            Default: {}
        stream_options: Optional[Dict[str, Any]]
            libCZI curl stream options for http(s) URLs, e.g. ``{"timeout": 60}`` or
            ``{"xoauth2_bearer": token}``. Ignored for local files.
            Default: None
        use_aicspylibczi: bool
            Deprecated and ignored. bioio-czi now always reads with the aicspylibczi
            library. Passing use_aicspylibczi=False raises, as the pylibczirw backend
            has been removed.
        """
        if use_aicspylibczi is not _USE_AICSPYLIBCZI_UNSET:
            if use_aicspylibczi is False:
                raise ValueError(
                    "The pylibczirw backend has been removed; use_aicspylibczi=False "
                    "is no longer supported. bioio-czi now always uses aicspylibczi."
                )
            warnings.warn(
                "use_aicspylibczi is deprecated and has no effect; bioio-czi always "
                "reads with aicspylibczi now.",
                DeprecationWarning,
                stacklevel=2,
            )

        # Expand details of provided image
        self._fs, self._path = io_utils.pathlike_to_fs(
            image,
            enforce_exists=True,
            fs_kwargs=fs_kwargs,
        )

        # Store params
        if isinstance(chunk_dims, str):
            chunk_dims = list(chunk_dims)

        self.chunk_dims = chunk_dims

        self._include_subblock_metadata = include_subblock_metadata
        self._stream_options = stream_options

        # Delayed storage
        self._px_sizes: Optional[types.PhysicalPixelSizes] = None
        self._mapped_dims: Optional[str] = None
        self._czi_scene_index: Optional[int] = None

        # Enforce valid image
        if not self._is_supported_image(
            self._fs, self._path, stream_options=stream_options
        ):
            raise exceptions.UnsupportedFileFormatError(
                self.__class__.__name__, self._path
            )

    @property
    def mapped_dims(self) -> str:
        if self._mapped_dims is None:
            with open_czi(self._fs, self._path, self._stream_options) as czi:
                self._mapped_dims = Reader._fix_czi_dims(czi.dims)

        return self._mapped_dims

    def _reset_self(self) -> None:
        super()._reset_self()
        self._mapped_dims = None
        self._px_sizes = None
        self._czi_scene_index = None
        self._dtype = None
        self.__dict__.pop("total_time_duration", None)

    @staticmethod
    def _fix_czi_dims(dims: str) -> str:
        return (
            dims.replace(CZI_BLOCK_DIM_CHAR, "")
            .replace(CZI_SCENE_DIM_CHAR, "")
            .replace(CZI_SAMPLES_DIM_CHAR, DimensionNames.Samples)
        )

    @property
    def scenes(self) -> Tuple[str, ...]:
        """Note: scenes with no name (`None`) will be renamed to
        "filename-<scene index>" to prevent ambiguity. Similarly, scenes with same
        names are automatically appended with occurrence number to distinguish
        between the two.

        Returns:
            Tuple[str, ...]: Scene names/id
        """
        if self._scenes is None:
            with open_czi(self._fs, self._path, self._stream_options) as czi:
                xpath_str = "./Metadata/Information/Image/Dimensions/S/Scenes/Scene"
                meta_scenes = czi.meta.findall(xpath_str)
                scene_names: List[str] = []

                # mapping of scene name to occurrences, indicating duplication.
                scene_name_frequency = {}
                for scene_idx, meta_scene in enumerate(meta_scenes):
                    shape = meta_scene.find("Shape")
                    if shape is not None:
                        shape_name = shape.get("Name")
                        scene_name = meta_scene.get("Name")
                        combined_scene_name = f"{scene_name}-{shape_name}"
                    else:
                        combined_scene_name = meta_scene.get("Name")
                        # Some scene names can be unpopulated, for those we should fill
                        # with filename-idx
                        if combined_scene_name is None:
                            fname_prefix = Path(self._path).stem
                            combined_scene_name = f"{fname_prefix}-{scene_idx}"
                        # Check for duplicated names
                        # first encounter with a duplicate modify original scene name
                        # to reflect its new duplicate status
                        if combined_scene_name not in scene_name_frequency:
                            scene_name_frequency[combined_scene_name] = [scene_idx, 1]
                        else:
                            if scene_name_frequency[combined_scene_name][1] == 1:
                                scene_names[
                                    scene_name_frequency[combined_scene_name][0]
                                ] += "-1"

                            scene_name_frequency[combined_scene_name][1] += 1

                            combined_scene_name += (
                                f"-{scene_name_frequency[combined_scene_name][1]}"
                            )

                    scene_names.append(combined_scene_name)

                # If the scene is implicit just assign it name Scene:0
                if len(scene_names) < 1:
                    scene_names = [metadata_utils.generate_ome_image_id(0)]
                else:
                    # reconcile scene list against the dims shape
                    dims_shape = czi.get_dims_shape()
                    if len(scene_names) != len(dims_shape) and czi.shape_is_consistent:
                        dims_shape_dict = dims_shape[0]
                        scene_range = dims_shape_dict.get(CZI_SCENE_DIM_CHAR)
                        if scene_range is not None:
                            scene_names = scene_names[scene_range[0] : scene_range[1]]
                        else:
                            # If this is the root node of a split multiscene czi,
                            # then the scene_range could be None because the dims_shape
                            # will be effectively empty.
                            # We do not currently support loading multi-file split
                            # scene CZI files
                            log.warning(
                                "CZI file appears to contain multiple scenes but "
                                "dimension data is not available in this file. "
                                "Root node of split multi-scene CZI files are not "
                                "supported by Reader."
                            )

                self._scenes = tuple(scene_names)

        return self._scenes

    @staticmethod
    def _dims_shape_to_scene_dims_shape(
        dims_shape: List[Dict], scene_index: int, consistent: bool
    ) -> Dict[str, Tuple[int, int]]:
        """
        This function takes the output of `get_dims_shape()` and returns a
        dictionary of dimensions for the selected scene

        Parameters
        ----------
        dims_shape: List[Dict]
            a list of dictionaries, generated by `get_dims_shape()`
        scene_index: int
            the index of the scene being used
        consistent: bool
            true if the dictionaries are consistent could be represented
            compactly (dims_shape with length 1)

        Returns
        -------
        A dictionary of dimensions, ie
        {"T": (0, 1), "C": (0, 3), "Y": (0, 256), "X":(0, 256)}.
        """
        dims_shape_index = 0 if consistent else scene_index
        dims_shape_dict = dims_shape[dims_shape_index]
        dims_shape_dict.pop(CZI_SCENE_DIM_CHAR, None)
        return dims_shape_dict

    @staticmethod
    def _adjust_scene_index(
        dims_shape: List[Dict], scene_index: int, consistent: bool
    ) -> int:
        """
        This function modifies a scene index to be an offset into the true scene
        indices reported by the czi file

        Parameters
        ----------
        dims_shape: List[Dict]
            a list of dictionaries, generated by `get_dims_shape()`
        scene_index: int
            the index of the scene being used
        consistent: bool
            true if the dictionaries are consistent could be represented
            compactly (dims_shape with length 1)

        Returns
        -------
        An int representing the scene index to use in a true libCZI dimension
        """
        dims_shape_index = 0 if consistent else scene_index
        dims_shape_dict = dims_shape[dims_shape_index]
        scene_range = dims_shape_dict.get(CZI_SCENE_DIM_CHAR)
        if scene_range is None:
            return scene_index
        if not consistent:
            # we have selected a dims_shape_dict already based on scene index
            # let's make sure the scene index is in the S range
            if scene_index < scene_range[0] or scene_index >= scene_range[1]:
                raise ValueError(
                    f"Scene index {scene_index} is not in the range "
                    f"{scene_range[0]} to {scene_range[1]}"
                )
            return scene_index
        return scene_range[0] + scene_index

    @staticmethod
    def _read_chunk_from_image(
        fs: AbstractFileSystem,
        path: str,
        scene: int,
        read_dims: Optional[Dict[str, int]] = None,
        stream_options: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        return Reader._get_image_data(
            fs=fs,
            path=path,
            scene=scene,
            read_dims=read_dims,
            stream_options=stream_options,
        )[0]

    @staticmethod
    def _get_image_data(
        fs: AbstractFileSystem,
        path: str,
        scene: int,
        read_dims: Optional[Dict[str, int]] = None,
        stream_options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
        """
        Read and return the squeezed image data requested along with the dimension info
        that was read.

        Parameters
        ----------
        fs: AbstractFileSystem
            The file system to use for reading.
        path: str
            The path to the file to read.
        scene: int
            The scene index to pull the chunk from.
        read_dims: Optional[Dict[str, int]]
            The dimensions to read from the file as a dictionary of string to integer.
            Default: None (Read all data from the image)

        Returns
        -------
        chunk: np.ndarray
            The image chunk read as a numpy array.
        read_dimensions: List[Tuple[str, int]]]
            The dimension sizes that were returned from the read.
        """
        # Init czi and delegate to the shared single-plane read.
        with open_czi(fs, path, stream_options) as czi:
            return Reader._read_plane(czi, scene, read_dims)

    @staticmethod
    def _read_plane(
        czi: CziFile,
        scene: int,
        read_dims: Optional[Dict[str, int]] = None,
    ) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
        """
        Read one (sub-)plane from an already-open CziFile.

        ``read_dims`` uses CZI-native dim chars and absolute indices. Dims present
        in read_dims (plus the block dim) are dropped to a single index; any dim
        not given (Y, X, Samples, ...) is read in full.

        Parameters
        ----------
        czi: CziFile
            An open CziFile to read from.
        scene: int
            The BioIO scene index to pull the plane from.
        read_dims: Optional[Dict[str, int]]
            The dimensions to fix as a dictionary of CZI-native char to absolute
            index. Default: None (read all data from the image).

        Returns
        -------
        chunk: np.ndarray
            The image chunk read as a numpy array (fixed dims dropped).
        read_dimensions: List[Tuple[str, int]]
            The dimension info for the dims that remained in the chunk.
        """
        # Copy so we don't mutate the caller's dict when injecting the scene.
        read_dims = dict(read_dims) if read_dims else {}

        # Get current scene read dims
        adjusted_scene_index = Reader._adjust_scene_index(
            czi.get_dims_shape(), scene, czi.shape_is_consistent
        )
        read_dims[CZI_SCENE_DIM_CHAR] = adjusted_scene_index

        # Read image
        data, dims = czi.read_image(**read_dims)

        # Drop dims that shouldn't be provided back
        ops: List[Union[int, slice]] = []
        real_dims = []
        for dim_info in dims:
            # Expand dimension info
            dim, _ = dim_info

            # If the dim was provided in the read dims
            # we know a single plane for that dimension was requested so remove it
            if dim in read_dims or dim == CZI_BLOCK_DIM_CHAR:
                ops.append(0)

            # Otherwise just read the full slice
            else:
                ops.append(slice(None, None, None))
                real_dims.append(dim_info)

        # Convert ops and run getitem
        return data[tuple(ops)], real_dims

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
            order = self.mapped_dims
            with open_czi(self._fs, self._path, self._stream_options) as czi:
                dims_shape = Reader._dims_shape_to_scene_dims_shape(
                    czi.get_dims_shape(),
                    self.current_scene_index,
                    czi.shape_is_consistent,
                )
            dims_shape.pop(CZI_BLOCK_DIM_CHAR, None)
            shape = tuple(dims_shape[_BIOIO_TO_CZI_DIM.get(d, d)][1] for d in order)
            self._dims = Dimensions(dims=order, shape=shape)
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

    @property
    def dtype(self) -> np.dtype:
        """
        Data type of the image array's elements.

        Returns
        -------
        dtype: np.dtype
            Data-type of the image array's elements.
        """
        if self._dtype is None:
            with open_czi(self._fs, self._path, self._stream_options) as czi:
                pixel_type = PIXEL_DICT.get(czi.pixel_type)
                if pixel_type is None:
                    raise TypeError(
                        f"Unsupported or unlabeled pixel type: {czi.pixel_type!r}"
                    )
            self._dtype = np.dtype(pixel_type)
        return self._dtype

    def _read_indexed(self, given_dims: str, dim_specs: list) -> np.ndarray:
        """
        Return the native-order array with ``dim_specs`` applied.

        This lets ``get_image_data`` read only the requested sub-region. It
        reads each plane one at a time via ``_read_plane`` (which fetches only
        the requested sub-blocks at the libCZI level), then crops the
        Y/X/Samples selection in memory.

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
        spatial = (
            DimensionNames.SpatialY,
            DimensionNames.SpatialX,
            DimensionNames.Samples,
        )
        cullable = [(i, d) for i, d in enumerate(given_dims) if d not in spatial]
        plane_specs = tuple(
            spec for d, spec in zip(given_dims, dim_specs) if d in spatial
        )

        with open_czi(self._fs, self._path, self._stream_options) as czi:
            dims_shape = Reader._dims_shape_to_scene_dims_shape(
                czi.get_dims_shape(),
                self.current_scene_index,
                czi.shape_is_consistent,
            )
            pixel_type = PIXEL_DICT.get(czi.pixel_type)
            if pixel_type is None:
                raise TypeError(
                    f"Unsupported or unlabeled pixel type: {czi.pixel_type!r}"
                )

            # Resolve each cullable dim to (czi_char, [(out_pos|None, abs_idx)...]).
            # read_image wants absolute CZI indices: begin + position.
            enumerated: List[Tuple[str, list]] = []
            kept_lengths: List[int] = []
            for i, d in cullable:
                czi_char = _BIOIO_TO_CZI_DIM.get(d, d)
                begin = dims_shape[czi_char][0]
                size_i = native_shape[i]
                spec = dim_specs[i]
                if isinstance(spec, slice):
                    idxs = list(range(*spec.indices(size_i)))
                    enumerated.append(
                        (czi_char, [(p, begin + j) for p, j in enumerate(idxs)])
                    )
                    kept_lengths.append(len(idxs))
                elif isinstance(spec, list):
                    idxs = [j % size_i for j in spec]
                    enumerated.append(
                        (czi_char, [(p, begin + j) for p, j in enumerate(idxs)])
                    )
                    kept_lengths.append(len(idxs))
                else:  # int -> fixed, axis dropped
                    enumerated.append((czi_char, [(None, begin + int(spec) % size_i)]))

            out: Optional[np.ndarray] = None
            for combo in itertools.product(*(entries for _c, entries in enumerated)):
                out_pos = tuple(p for p, _idx in combo if p is not None)
                read_dims = {
                    czi_char: idx
                    for (czi_char, _entries), (_p, idx) in zip(enumerated, combo)
                }
                plane, _ = Reader._read_plane(czi, self.current_scene_index, read_dims)
                cropped = plane[plane_specs]
                if out is None:
                    out = np.empty(
                        tuple(kept_lengths) + cropped.shape, dtype=cropped.dtype
                    )
                out[out_pos] = cropped

        if out is None:
            # A dim selected nothing (e.g. C=slice(0, 0)) so the read
            # loop never ran; build a full-dimensionality empty result.
            spatial_full = tuple(
                native_shape[i] for i, d in enumerate(given_dims) if d in spatial
            )
            spatial_shape = np.empty(spatial_full)[plane_specs].shape
            out = np.empty(tuple(kept_lengths) + spatial_shape, dtype=pixel_type)
        return out

    def _create_dask_array(self, czi: CziFile) -> xr.DataArray:
        """
        Creates a delayed dask array for the file.

        Parameters
        ----------
        czi: CziFile
            An open CziFile for processing.

        Returns
        -------
        image_data: da.Array
            The fully constructed and fully delayed image as a Dask Array object.
        """
        # Always add the plane dimensions if not present already
        for dim in REQUIRED_CHUNK_DIMS:
            if dim not in self.chunk_dims:
                self.chunk_dims.append(dim)

        # Safety measure / "feature"
        self.chunk_dims = [d.upper() for d in self.chunk_dims]

        # Construct the delayed dask array
        dims_shape = Reader._dims_shape_to_scene_dims_shape(
            czi.get_dims_shape(),
            scene_index=self.current_scene_index,
            consistent=czi.shape_is_consistent,
        )

        # Remove block dim as not useful
        dims_shape.pop(CZI_BLOCK_DIM_CHAR, None)

        dims_str = czi.dims
        for remove_dim_char in [CZI_BLOCK_DIM_CHAR, CZI_SCENE_DIM_CHAR]:
            dims_str = dims_str.replace(remove_dim_char, "")

        # Get the shape for the chunk and operating shape for the dask array
        # We also collect the chunk and non chunk dimension ordering so that we can
        # swap the dimensions after we
        # block the dask array together.
        sample_chunk_shape = []
        operating_shape = []
        non_chunk_dimension_ordering = []
        chunk_dimension_ordering = []
        for i, dim in enumerate(dims_str):
            # Unpack dim info
            _, dim_size = dims_shape[dim]

            # If the dim is part of the specified chunk dims then append it to the
            # sample, and, append the dimension
            # to the chunk dimension ordering
            if dim in self.chunk_dims:
                sample_chunk_shape.append(dim_size)
                chunk_dimension_ordering.append(dim)

            # Otherwise, append the dimension to the non chunk dimension ordering, and,
            # append the true size of the
            # image at that dimension
            else:
                non_chunk_dimension_ordering.append(dim)
                operating_shape.append(dim_size)

        # Convert shapes to tuples and combine the non and chunked dimension orders as
        # that is the order the data will
        # actually come out of the read data as
        sample_chunk_shape_tuple = tuple(sample_chunk_shape)
        blocked_dimension_order = (
            non_chunk_dimension_ordering + chunk_dimension_ordering
        )

        # Fill out the rest of the operating shape with dimension sizes of 1 to match
        # the length of the sample chunk
        # When dask.block happens it fills the dimensions from inner-most to outer-most
        # with the chunks as long as the dimension is size 1
        # Basically, we are adding empty dimensions to the operating shape that will be
        # filled by the chunks from dask
        operating_shape_tuple = tuple(operating_shape) + (1,) * len(
            sample_chunk_shape_tuple
        )

        # Create empty numpy array with the operating shape so that we can iter through
        # and use the multi_index to create the readers.
        lazy_arrays: np.ndarray = np.ndarray(operating_shape_tuple, dtype=object)

        # We can enumerate over the multi-indexed array and construct read_dims
        # dictionaries by simply zipping together the ordered dims list and the current
        # multi-index plus the begin index for that plane. We then set the value of the
        # array at the same multi-index to the delayed reader using the constructed
        # read_dims dictionary.
        dims = [
            d for d in czi.dims if d not in [CZI_BLOCK_DIM_CHAR, CZI_SCENE_DIM_CHAR]
        ]
        begin_indicies = tuple(dims_shape[d][0] for d in dims)
        for np_index, _ in np.ndenumerate(lazy_arrays):
            # Add the czi file begin index for each dimension to the array dimension
            # index
            this_chunk_read_indicies = (
                current_dim_begin_index + curr_dim_index
                for current_dim_begin_index, curr_dim_index in zip(
                    begin_indicies, np_index
                )
            )

            # Zip the dims with the read indices
            this_chunk_read_dims = dict(
                zip(blocked_dimension_order, this_chunk_read_indicies)
            )

            # Remove the dimensions that we want to chunk by from the read dims
            for d in self.chunk_dims:
                this_chunk_read_dims.pop(d, None)

            # Get pixel type and catch unsupported
            pixel_type = PIXEL_DICT.get(czi.pixel_type)
            if pixel_type is None:
                raise TypeError(
                    f"Unsupported or unlabeled pixel type: {czi.pixel_type!r}"
                )

            # Add delayed array to lazy arrays at index
            lazy_arrays[np_index] = da.from_delayed(
                delayed(Reader._read_chunk_from_image)(
                    fs=self._fs,
                    path=self._path,
                    scene=self.current_scene_index,
                    read_dims=this_chunk_read_dims,
                    stream_options=self._stream_options,
                ),
                shape=sample_chunk_shape,
                dtype=pixel_type,
            )

        # Convert the numpy array of lazy readers into a dask array and fill the inner
        # most empty dimensions with chunks
        merged = da.block(lazy_arrays.tolist())

        # Because we have set certain dimensions to be chunked and others not
        # we will need to transpose back to original dimension ordering
        # Example being, if the original dimension ordering was "SZYX" and we want to
        # chunk by "S", "Y", and "X" we created an array with dimensions ordering "ZSYX"
        transpose_indices = []
        transpose_required = False
        for i, d in enumerate(dims_str):
            new_index = blocked_dimension_order.index(d)
            if new_index != i:
                transpose_required = True
                transpose_indices.append(new_index)
            else:
                transpose_indices.append(i)

        # Only run if the transpose is actually required
        # The default case is "Z", "Y", "X", which _usually_ doesn't need to be
        # transposed because that is _usually_ the normal dimension order of the CZI
        # file anyway
        if transpose_required:
            merged = da.transpose(merged, tuple(transpose_indices))

        # Because dimensions outside of Y and X can be in any order and present or not
        # we also return the dimension order string.
        return merged

    @staticmethod
    def _get_coords_and_physical_px_sizes(
        xml: ET.Element, scene_index: int, dims_shape: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], types.PhysicalPixelSizes]:
        # Create coord dict
        coords: Dict[str, Any] = {}

        # Attach channel names to coords
        scene_channel_list = get_channel_names(xml, scene_index, dims_shape)
        if scene_channel_list is not None:
            coords[DimensionNames.Channel] = scene_channel_list

        # Handle Spatial Dimensions
        px_sizes = get_physical_pixel_sizes(xml)
        for dim_name, scale in px_sizes._asdict().items():
            if scale is not None and dim_name in dims_shape:
                dim_size = size(dims_shape, dim_name)
                coords[dim_name] = Reader._generate_coord_array(0, dim_size, scale)

        return coords, px_sizes

    def _read_delayed(self) -> xr.DataArray:
        """
        Construct the delayed xarray DataArray object for the image.

        Returns
        -------
        image: xr.DataArray
            The fully constructed and fully delayed image as a DataArray object.
            Metadata is attached in some cases as coords, dims, and attrs.

        Raises
        ------
        exceptions.UnsupportedFileFormatError
            The file could not be read or is not supported.
        """
        with open_czi(self._fs, self._path, self._stream_options) as czi:

            dims_shape = Reader._dims_shape_to_scene_dims_shape(
                dims_shape=czi.get_dims_shape(),
                scene_index=self.current_scene_index,
                consistent=czi.shape_is_consistent,
            )

            # Get dims as list for xarray
            img_dims_list = list(self.mapped_dims)

            # Get image data
            image_data = self._create_dask_array(czi)

            # Create coordinate planes
            meta = czi.meta
            coords, px_sizes = self._get_coords_and_physical_px_sizes(
                xml=meta,
                scene_index=self.current_scene_index,
                dims_shape=dims_shape,
            )

            # Append subblock metadata to the other metadata if param is True
            if self._include_subblock_metadata:
                subblocks = czi.read_subblock_metadata(unified_xml=True)
                meta.append(subblocks)

            # Store pixel sizes
            self._px_sizes = px_sizes

            # handle edge case where image has 0,0 YX dims:
            if image_data.shape[-2:] == (0, 0):
                return xr.DataArray(
                    dims=coords.keys(),
                    coords=coords,
                    attrs={constants.METADATA_UNPROCESSED: meta},
                )
            else:
                return xr.DataArray(
                    image_data,
                    dims=img_dims_list,
                    coords=coords,
                    attrs={constants.METADATA_UNPROCESSED: meta},
                )

    def _read_immediate(self) -> xr.DataArray:
        """
        Construct the in-memory xarray DataArray object for the image.

        Returns
        -------
        image: xr.DataArray
            The fully constructed and fully read into memory image as a DataArray
            object. Metadata is attached in some cases as coords, dims, and attrs.

        Raises
        ------
        exceptions.UnsupportedFileFormatError
            The file could not be read or is not supported.
        """
        with open_czi(self._fs, self._path, self._stream_options) as czi:
            dims_shape = Reader._dims_shape_to_scene_dims_shape(
                dims_shape=czi.get_dims_shape(),
                scene_index=self.current_scene_index,
                consistent=czi.shape_is_consistent,
            )

            # Get image data
            image_data, _ = self._get_image_data(
                fs=self._fs,
                path=self._path,
                scene=self.current_scene_index,
                stream_options=self._stream_options,
            )

            # Get metadata
            meta = czi.meta

            # Create coordinate planes
            coords, px_sizes = self._get_coords_and_physical_px_sizes(
                xml=meta,
                scene_index=self.current_scene_index,
                dims_shape=dims_shape,
            )

            # Store pixel sizes
            self._px_sizes = px_sizes

            return xr.DataArray(
                image_data,
                dims=[d for d in self.mapped_dims],
                coords=coords,
                attrs={constants.METADATA_UNPROCESSED: meta},
            )

    @staticmethod
    def _read_mosaic_region(
        fs: AbstractFileSystem,
        path: str,
        region: Tuple[int, int, int, int],
        shape: Tuple[int, ...],
        read_dims: Dict[str, int],
        stream_options: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        """
        Composite one plane of the stitched mosaic over ``region`` (x, y, w, h).
        libCZI reads only the tiles intersecting the region.
        """
        with open_czi(fs, path, stream_options) as czi:
            data = czi.read_mosaic(region=region, scale_factor=1.0, **read_dims)
        return data.reshape(shape)

    def _mosaic_planes(
        self, czi: CziFile
    ) -> Tuple[Dict[Tuple[int, ...], Dict[str, int]], Tuple[int, ...], BBox]:
        """
        Returns the absolute CZI read dims of every plane in the current scene keyed
        by its index in the non-tile dims, the shape of those dims, and the scene's
        bounding box within the mosaic.
        """
        dims_shape = Reader._dims_shape_to_scene_dims_shape(
            czi.get_dims_shape(), self.current_scene_index, czi.shape_is_consistent
        )
        plane_dims = [
            d
            for d in self.mapped_dims
            if d
            not in (
                DimensionNames.MosaicTile,
                DimensionNames.SpatialY,
                DimensionNames.SpatialX,
                DimensionNames.Samples,
            )
        ]
        shape = tuple(size(dims_shape, d) for d in plane_dims)
        planes = {
            index: {d: dims_shape[d][0] + i for d, i in zip(plane_dims, index)}
            for index in np.ndindex(shape)
        }
        bbox = czi.get_mosaic_scene_bounding_box(index=self.czi_scene_index)
        return planes, shape, bbox

    def _mosaic_plane_shape(self, height: int, width: int) -> Tuple[int, ...]:
        if DimensionNames.Samples in self.dims.order:
            return (height, width, self.dims.S)
        return (height, width)

    def _get_stitched_dask_mosaic(self) -> xr.DataArray:
        with open_czi(self._fs, self._path, self._stream_options) as czi:
            planes, shape, bbox = self._mosaic_planes(czi)

        # One chunk per native tile, so a window into the mosaic reads only the
        # tiles beneath it. A trailing singleton keeps da.block from concatenating
        # along the samples axis.
        tile_h, tile_w = self.dims.Y, self.dims.X
        y_starts = range(0, bbox.h, tile_h)
        x_starts = range(0, bbox.w, tile_w)
        grid = shape + (len(y_starts), len(x_starts))
        if DimensionNames.Samples in self.dims.order:
            grid += (1,)
        blocks: np.ndarray = np.ndarray(grid, dtype=object)
        for index in np.ndindex(blocks.shape):
            plane_index = index[: len(shape)]
            y0 = y_starts[index[len(shape)]]
            x0 = x_starts[index[len(shape) + 1]]
            chunk_shape = self._mosaic_plane_shape(
                min(tile_h, bbox.h - y0), min(tile_w, bbox.w - x0)
            )
            blocks[index] = da.from_delayed(
                delayed(Reader._read_mosaic_region)(
                    self._fs,
                    self._path,
                    (bbox.x + x0, bbox.y + y0, chunk_shape[1], chunk_shape[0]),
                    chunk_shape,
                    planes[plane_index],
                    self._stream_options,
                ),
                shape=chunk_shape,
                dtype=self.dtype,
            )
        return self._construct_mosaic_xarray(da.block(blocks.tolist()))

    def _get_stitched_mosaic(self) -> xr.DataArray:
        with open_czi(self._fs, self._path, self._stream_options) as czi:
            planes, shape, bbox = self._mosaic_planes(czi)
        plane_shape = self._mosaic_plane_shape(bbox.h, bbox.w)
        stitched = np.empty(shape + plane_shape, dtype=self.dtype)
        for index, read_dims in planes.items():
            stitched[index] = Reader._read_mosaic_region(
                self._fs,
                self._path,
                (bbox.x, bbox.y, bbox.w, bbox.h),
                plane_shape,
                read_dims,
                self._stream_options,
            )
        return self._construct_mosaic_xarray(stitched)

    def _construct_mosaic_xarray(self, stitched: types.ArrayLike) -> xr.DataArray:
        dims = [
            d for d in self.xarray_dask_data.dims if d is not DimensionNames.MosaicTile
        ]
        coords: Dict[Hashable, Any] = {
            d: v
            for d, v in self.xarray_dask_data.coords.items()
            if d
            not in [
                DimensionNames.MosaicTile,
                DimensionNames.SpatialY,
                DimensionNames.SpatialX,
            ]
        }

        # Add expanded Y and X coords
        if self.physical_pixel_sizes.Y is not None:
            dim_y_index = dims.index(DimensionNames.SpatialY)
            coords[DimensionNames.SpatialY] = Reader._generate_coord_array(
                0, stitched.shape[dim_y_index], self.physical_pixel_sizes.Y
            )
        if self.physical_pixel_sizes.X is not None:
            dim_x_index = dims.index(DimensionNames.SpatialX)
            coords[DimensionNames.SpatialX] = Reader._generate_coord_array(
                0, stitched.shape[dim_x_index], self.physical_pixel_sizes.X
            )

        return xr.DataArray(
            data=stitched,
            dims=dims,
            coords=coords,
            attrs=copy(self.xarray_dask_data.attrs),
        )

    @property
    def czi_scene_index(self) -> int:
        """
        Returns the plate-level CZI scene index for the current scene.

        For split CZI files (one plate position per file), this differs from
        current_scene_index (which is always 0) because the embedded XML metadata
        still uses the original plate-wide scene indices.
        """
        if self._czi_scene_index is None:
            with open_czi(self._fs, self._path, self._stream_options) as czi:
                self._czi_scene_index = Reader._adjust_scene_index(
                    czi.get_dims_shape(),
                    self.current_scene_index,
                    czi.shape_is_consistent,
                )
        return self._czi_scene_index

    @property
    def physical_pixel_sizes(self) -> types.PhysicalPixelSizes:
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
        if self._px_sizes is None:
            # We get pixel sizes as a part of array construct
            # so simply run array construct
            self.dask_data

        if self._px_sizes is None:
            raise ValueError("Pixel sizes weren't created as a part of image reading")

        return self._px_sizes

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
        if DimensionNames.MosaicTile not in self.dims.order:
            raise exceptions.UnexpectedShapeError("No mosaic dimension in image.")

        with open_czi(self._fs, self._path, self._stream_options) as czi:

            # Default Channel and Time dimensions to 0 to improve
            # worst case read time for large files **only**
            # when those dimensions are present on the image.
            for dimension_name in [DimensionNames.Channel, DimensionNames.Time]:
                if dimension_name not in kwargs and dimension_name in self.dims.order:
                    kwargs[dimension_name] = 0

            bbox = czi.get_mosaic_tile_bounding_box(
                M=mosaic_tile_index, S=self.czi_scene_index, **kwargs
            )
            return bbox.y, bbox.x

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
        if DimensionNames.MosaicTile not in self.dims.order:
            raise exceptions.UnexpectedShapeError("No mosaic dimension in image.")

        with open_czi(self._fs, self._path, self._stream_options) as czi:

            tile_info_to_bboxes = czi.get_all_mosaic_tile_bounding_boxes(
                S=self.czi_scene_index, **kwargs
            )

            # Convert dictionary of tile info mappings to
            # a list of bounding boxes sorted according to their
            # respective M indexes
            m_indexes_to_mosaic_positions = {
                tile_info.m_index: (bbox.y, bbox.x)
                for tile_info, bbox in tile_info_to_bboxes.items()
            }
            return [
                m_indexes_to_mosaic_positions[m_index]
                for m_index in sorted(m_indexes_to_mosaic_positions.keys())
            ]

    @property
    def acquisition_times(self) -> Optional[list[dict[str, int | datetime]]]:
        """
        Return the earliest acquisition time for each mosaic tile and timepoint.

        Returns
        -------
        Optional[list[dict[str, int | datetime]]]:
            A list of dictionaries, each containing subblock info and the corresponding
            acquisition time under the key "acquisition_time".
            Returns None if extraction fails.
        """

        with open_czi(self._fs, self._path, self._stream_options) as czi:
            return acquisition_times(
                czi=czi,
                current_scene=self.czi_scene_index,
            )

    def get_subblock_metadata(self, **kwargs: int) -> ET.Element:
        """
        Read the metadata of the current scene's subblocks matching the given
        dimension indices, e.g. ``T=0, C=1`` or ``M=3``, as a single ``Subblocks``
        element. Only the matching subblocks are read from the file.
        """
        if "S" in kwargs:
            raise ValueError("Select the scene with set_scene rather than S.")
        with open_czi(self._fs, self._path, self._stream_options) as czi:
            return czi.read_subblock_metadata(
                unified_xml=True, S=self.czi_scene_index, **kwargs
            )

    @property
    def time_interval(self) -> Optional[timedelta]:
        """
        Extracts the the average time interval between consecutive timepoints
        as a timedelta object.

        Returns
        -------
        Optional[timedelta]
            Average interval between timepoints.
            Returns None if total_time_duration is None or less than two timepoints.
        """

        # The purpose of this conditional is to not log a warning for files with a
        # single timepoint.
        timepoints = (
            self.dims[DimensionNames.Time][0]
            if DimensionNames.Time in self.dims.order
            else None
        )
        if timepoints is None or timepoints < 2:
            return None

        total_duration = self.total_time_duration
        if total_duration is None:
            return None
        return total_duration / (timepoints - 1)

    @functools.cached_property
    def total_time_duration(self) -> Optional[timedelta]:
        """
        Extracts the total duration of the timelapse as a timedelta object.
        This is the time between the first acquisition and the first acquisition of the
        last timepoint.

        Returns
        -------
        Optional[timedelta]
            Total time duration as a timedelta object.
            Returns None if extraction fails.
        """
        timepoints = (
            self.dims[DimensionNames.Time][0]
            if DimensionNames.Time in self.dims.order
            else None
        )
        if timepoints is None or timepoints < 2:
            return None

        try:
            with open_czi(self._fs, self._path, self._stream_options) as czi:
                duration_ms = time_between_subblocks(
                    czi,
                    self.czi_scene_index,
                    start_frame=0,
                    # Index of the last timepoint is one less than the number of
                    # timepoints
                    end_frame=timepoints - 1,
                )
                return (
                    timedelta(milliseconds=duration_ms)
                    if duration_ms is not None
                    else None
                )

        except Exception as exc:
            log.warning("Failed to extract Total Time Duration: %s", exc, exc_info=True)

        return None

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
        ome = metadata_utils.transform_metadata_with_xslt(
            self.metadata,
            Path(__file__).parent / "czi-to-ome-xslt/xslt/czi-to-ome.xsl",
        )

        # NOTE:
        # The OME metadata generated via XSLT reflects the raw CZI XML, which may
        # describe the original acquisition frame dimensions. However, the actual
        # pixel data exposed by this reader is derived from libCZI bounding boxes
        # (e.g., scene-specific ROI, stitching, and no-pyramid extents), which can
        # differ from the XML-reported sizes.
        #
        # This can lead to mismatches between `reader.shape` and
        # `ome_metadata.images[0].pixels.{size_x, size_y}`, particularly for
        # mosaics or scenes with adjusted bounding regions.
        #
        # To ensure consistency and correctness for downstream consumers, we
        # normalize the OME Pixel sizes to match the dimensions of the data
        # actually returned by the reader.
        if not ome.images:
            return ome

        pixels = ome.images[0].pixels
        dim_to_size = dict(zip(self.dims.order, self.shape))

        if "X" in dim_to_size:
            pixels.size_x = dim_to_size["X"]
        if "Y" in dim_to_size:
            pixels.size_y = dim_to_size["Y"]
        if "Z" in dim_to_size:
            pixels.size_z = dim_to_size["Z"]
        if "C" in dim_to_size:
            pixels.size_c = dim_to_size["C"]
        if "T" in dim_to_size:
            pixels.size_t = dim_to_size["T"]

        return ome

    @property
    def standard_metadata(self) -> StandardMetadata:
        """
        Return the standard metadata for this reader, updating specific fields.
        This implementation calls the base reader's standard_metadata property
        via super() and then assigns the new values.
        """
        # 1. Some of the standard metadata can be read from all bioio Readers in the
        # same way, which the following super() call does (e.g. binning, objective).
        metadata = super().standard_metadata

        # 2. The self-contained standard_metadata module holds the logic for
        # extracting these fields from the metadata.
        czi_scene_index = self.czi_scene_index
        metadata.column = standard_metadata_utils.column(self.metadata, czi_scene_index)
        metadata.position_index = standard_metadata_utils.position_index(
            self.current_scene
        )
        metadata.row = standard_metadata_utils.row(self.metadata, czi_scene_index)
        metadata.stage_position_x, metadata.stage_position_y = (
            standard_metadata_utils.scene_stage_position(self.metadata, czi_scene_index)
        )
        # 3. Override timelapse_interval and total_time_duration with the
        # subblock-derived values: super() sets timelapse_interval from the OME
        # transform, but the CZI subblock timing (self.time_interval) is preferred.
        metadata.timelapse_interval = self.time_interval
        metadata.total_time_duration = self.total_time_duration

        return metadata
