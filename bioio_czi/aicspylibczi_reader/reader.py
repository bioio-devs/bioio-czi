import itertools
import logging
import threading
import time
import xml.etree.ElementTree as ET
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Generator, Hashable, List, Optional, Tuple, Union

import dask.array as da
import numpy as np
import xarray as xr
from _aicspylibczi import BBox
from aicspylibczi import CziFile, remote_reads_available
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
from dask import delayed
from fsspec.implementations.local import LocalFileSystem
from fsspec.spec import AbstractFileSystem

from .. import metadata as metadata_utils
from .. import remote
from ..bounding_box import size
from ..channels import get_channel_names
from ..pixel_sizes import get_physical_pixel_sizes
from .subblock_metadata import acquisition_times, time_between_subblocks

###############################################################################

log = logging.getLogger(__name__)

READER_NAME = "bioio-czi[aicspylibczi mode]"

###############################################################################

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


@dataclass
class CziSource:
    """
    Everything needed to open one CZI, wherever it lives.

    aicspylibczi holds no file handle between calls, so the reader reopens the CZI
    for each piece of work it does. This bundles the "how to reopen it" details so
    they travel together, including into dask graphs -- which is why it stores a
    filesystem and a path rather than an open handle.

    Remote handles are cached on the source so that successive reads (metadata, image
    data, mosaic planes) share a single open connection without re-fetching the header
    and sub-block directory each time. The cache is a non-picklable attribute: when
    the source is serialised into a dask graph and unpickled in a worker process, the
    cache starts empty and the worker builds its own handle independently.

    Parameters
    ----------
    fs: AbstractFileSystem
        The filesystem holding the image.
    path: str
        The protocol-stripped path within ``fs``, e.g. "bucket/key" for
        "s3://bucket/key".
    uri: str
        The image location with its protocol intact. Presigning needs this, because
        ``path`` has had the protocol stripped off by fsspec.
    stream_options: Optional[Dict[str, Any]]
        libCZI stream options, e.g. ``{"timeout": 30}``. Remote sources only.
        Default: None
    url_expiration: int
        Seconds a generated presigned URL stays valid.
        Default: remote.DEFAULT_URL_EXPIRATION_SECONDS
    """

    fs: AbstractFileSystem
    path: str
    uri: str
    stream_options: Optional[Dict[str, Any]] = None
    url_expiration: int = remote.DEFAULT_URL_EXPIRATION_SECONDS

    def __post_init__(self) -> None:
        self._handle: Optional[CziFile] = None
        self._handle_opened_at: float = 0.0
        self._lock = threading.Lock()

    def __getstate__(self) -> Dict[str, Any]:
        return {
            "fs": self.fs,
            "path": self.path,
            "uri": self.uri,
            "stream_options": self.stream_options,
            "url_expiration": self.url_expiration,
        }

    def __setstate__(self, state: Dict[str, Any]) -> None:
        for k, v in state.items():
            object.__setattr__(self, k, v)
        self.__post_init__()

    @classmethod
    def from_fs(
        cls,
        fs: AbstractFileSystem,
        path: str,
        **kwargs: Any,
    ) -> "CziSource":
        """
        Build a source from the ``(fs, path)`` pair BioIO readers are handed.

        The protocol is put back onto ``path`` so that object-store paths can be
        presigned later. Local paths are left alone rather than turned into
        ``file://`` URIs.
        """
        if isinstance(fs, LocalFileSystem) or remote.uri_scheme(path):
            # A path that still carries its scheme -- which is what fsspec hands
            # back for http(s) -- is already a URI. Putting the protocol back on
            # anyway would double it up, e.g. "https://http://host/a.czi".
            uri = path
        else:
            uri = fs.unstrip_protocol(path)
        return cls(fs=fs, path=path, uri=uri, **kwargs)

    @property
    def is_remote(self) -> bool:
        """
        Whether this CZI is read over the network rather than off local disk.
        """
        return remote.is_remote(self.uri)

    @property
    def needs_signing(self) -> bool:
        """
        Whether reaching this image means generating a presigned URL first.
        """
        return self.is_remote and not remote.is_http_url(self.uri)

    def _open_remote(self) -> CziFile:
        """
        Open the CZI over the network, presigning its URI if it needs it.
        """
        if remote.is_http_url(self.uri):
            url = self.uri
        else:
            url = remote.sign_url(
                self.fs,
                self.path,
                uri=self.uri,
                reader_name=READER_NAME,
                expiration=self.url_expiration,
            )
        return CziFile(url, stream_options=self.stream_options or None)

    def _handle_is_stale(self) -> bool:
        if not self.needs_signing:
            return False
        return (time.monotonic() - self._handle_opened_at) >= 0.9 * self.url_expiration

    @contextmanager
    def open(self) -> Generator[CziFile, None, None]:
        """
        Open the CZI, yielding a ``CziFile`` for the duration of the block.

        Remote handles are cached and reused, because reopening one means refetching
        the header, metadata and sub-block directory over the network before any
        pixels can be read. Local handles are not cached: reopening a local file is
        immeasurably cheap next to a read, and holding the descriptor open would keep
        the file locked for as long as the process lives.

        Raises
        ------
        exceptions.UnsupportedFileFormatError
            The image is remote but this aicspylibczi build cannot read remote
            files, or its protocol cannot be turned into an http(s) URL.
        """
        if self.is_remote:
            require_remote_reads(self.uri)
            with self._lock:
                if self._handle is None or self._handle_is_stale():
                    self._handle = self._open_remote()
                    self._handle_opened_at = time.monotonic()
                handle = self._handle
            yield handle
        else:
            with self.fs.open(self.path) as open_resource:
                yield CziFile(open_resource.f)


def require_remote_reads(uri: str) -> None:
    """
    Raise unless this aicspylibczi installation can read CZIs over http(s).

    Remote reads need libCZI's curl-based stream, which is a build-time option in
    aicspylibczi, so a working import is not enough to know they are available.

    Raises
    ------
    exceptions.UnsupportedFileFormatError
        This build was compiled without the curl stream.
    """
    if not remote_reads_available():
        raise exceptions.UnsupportedFileFormatError(
            READER_NAME,
            uri,
            "This aicspylibczi build was compiled without libCZI's curl stream, so "
            "it cannot read CZIs over the network. Reinstall aicspylibczi from a "
            "wheel built with remote support, or drop use_aicspylibczi to read this "
            "image in pylibczirw mode.",
        )


class Reader(BaseReader):
    """
    Wraps the aicspylibczi API to provide the same BioIO Reader plugin for
    volumetric Zeiss CZI images.

    Notes
    -----
    To use this reader, install with: `pip install aicspylibczi>=4.0.0`.
    """

    NAME = "bioio-czi-aicspylibczi"

    _xarray_dask_data: Optional["xr.DataArray"] = None
    _xarray_data: Optional["xr.DataArray"] = None
    _mosaic_xarray_dask_data: Optional["xr.DataArray"] = None
    _mosaic_xarray_data: Optional["xr.DataArray"] = None
    _dims: Optional[Dimensions] = None
    _metadata: Optional[Any] = None
    _scenes: Optional[Tuple[str, ...]] = None
    _current_scene_index: int = 0
    # Do not provide default value because
    # they may not need to be used by your reader (i.e. input param is an array)
    _fs: "AbstractFileSystem"
    _path: str
    # How to reopen the image; see CziSource. _fs and _path are kept alongside it
    # because BioIO (and its test utilities) expect every reader to expose them.
    _source: CziSource

    @staticmethod
    def _is_supported_image(fs: AbstractFileSystem, path: str, **kwargs: Any) -> bool:
        source = CziSource.from_fs(
            fs,
            path,
            stream_options=kwargs.get("stream_options"),
        )
        try:
            with source.open():
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
                source.uri,
                str(e),
            )

    def __init__(
        self,
        image: types.PathLike,
        chunk_dims: Union[str, List[str]] = DEFAULT_CHUNK_DIMS,
        include_subblock_metadata: bool = False,
        fs_kwargs: Dict[str, Any] = {},
        stream_options: Optional[Dict[str, Any]] = None,
        url_expiration: int = remote.DEFAULT_URL_EXPIRATION_SECONDS,
        mosaic_chunk_size: Optional[Tuple[int, int]] = None,
    ):
        """
        Parameters
        ----------
        image: types.PathLike
            Path to image file to construct Reader for. May be a local path, an
            http(s) URL, or an object-store URI such as "s3://bucket/key". See the
            Notes section for how remote images are read.
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
            Default: {}
        stream_options: Optional[Dict[str, Any]]
            libCZI stream options for remote images, e.g. ``{"timeout": 60}`` or
            ``{"xoauth2_bearer": token}``. Ignored for local images.
            Default: None
        url_expiration: int
            How long, in seconds, a presigned URL generated for an object-store
            image stays valid. Only relevant for protocols that must be presigned,
            e.g. "s3://". Reads are issued for as long as this reader (and any dask
            graph built from it) is in use, so this needs to outlast a read session.
            Default: remote.DEFAULT_URL_EXPIRATION_SECONDS
        mosaic_chunk_size: Optional[Tuple[int, int]]
            The (height, width) of the chunks the stitched mosaic is read in, which is
            the granularity at which a window into it costs anything. The default of
            one native tile per chunk makes small windows as cheap as possible.
            Because tiles overlap, a tile-sized grid re-reads some sub-blocks when the
            whole mosaic is pulled through the dask path, so code that always reads
            entire mosaics can pass a larger size to trade window latency for fewer
            reads.
            Default: None (one native tile per chunk)

        Notes
        -----
        Remote images are read by libCZI's curl-based stream, which fetches only the
        byte ranges it needs rather than downloading the whole file. Protocols other
        than http(s) are presigned into an https URL by their fsspec filesystem, so
        credentials are resolved by fsspec in the usual way and never handed to
        libCZI. This requires an aicspylibczi built with remote support; see
        :func:`require_remote_reads`.
        """
        # Expand details of provided image
        self._fs, self._path = io_utils.pathlike_to_fs(
            image,
            enforce_exists=True,
            fs_kwargs=fs_kwargs,
        )

        self._source = CziSource.from_fs(
            self._fs,
            self._path,
            stream_options=stream_options,
            url_expiration=url_expiration,
        )

        # Store params
        if isinstance(chunk_dims, str):
            chunk_dims = list(chunk_dims)

        self.chunk_dims = chunk_dims

        self._include_subblock_metadata = include_subblock_metadata

        self._mosaic_chunk_size = mosaic_chunk_size

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
            with self._source.open() as czi:
                self._mapped_dims = Reader._fix_czi_dims(czi.dims)

        return self._mapped_dims

    def _reset_self(self) -> None:
        super()._reset_self()
        self._mapped_dims = None
        self._px_sizes = None
        self._czi_scene_index = None

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
            with self._source.open() as czi:
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
        source: CziSource,
        scene: int,
        read_dims: Optional[Dict[str, int]] = None,
    ) -> np.ndarray:
        return Reader._get_image_data(source=source, scene=scene, read_dims=read_dims)[
            0
        ]

    @staticmethod
    def _get_image_data(
        source: CziSource,
        scene: int,
        read_dims: Optional[Dict[str, int]] = None,
    ) -> Tuple[np.ndarray, List[Tuple[str, int]]]:
        """
        Read and return the squeezed image data requested along with the dimension info
        that was read.

        Parameters
        ----------
        source: CziSource
            Where the image lives and how to reopen it.
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
        with source.open() as czi:
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
            with self._source.open() as czi:
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

        with self._source.open() as czi:
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
                    source=self._source,
                    scene=self.current_scene_index,
                    read_dims=this_chunk_read_dims,
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
        with self._source.open() as czi:

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
        with self._source.open() as czi:
            dims_shape = Reader._dims_shape_to_scene_dims_shape(
                dims_shape=czi.get_dims_shape(),
                scene_index=self.current_scene_index,
                consistent=czi.shape_is_consistent,
            )

            # Get image data
            image_data, _ = self._get_image_data(
                source=self._source,
                scene=self.current_scene_index,
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

    def _construct_mosaic_xarray(self, stitched: types.ArrayLike) -> xr.DataArray:
        """
        Wrap an already-stitched mosaic array in the metadata of this scene.

        Parameters
        ----------
        stitched: types.ArrayLike
            The stitched mosaic, in this reader's dimension order minus the mosaic
            tile dimension. Either in memory or a dask array.
        """
        # Copy metadata
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

        attrs = copy(self.xarray_dask_data.attrs)

        return xr.DataArray(
            data=stitched,
            dims=dims,
            coords=coords,
            attrs=attrs,
        )

    @staticmethod
    def _read_mosaic_region(
        source: CziSource,
        read_dims: Dict[str, int],
        region: Tuple[int, int, int, int],
        out_shape: Tuple[int, ...],
        dtype: np.dtype,
    ) -> np.ndarray:
        """
        Read one rectangle of the stitched mosaic, composited by libCZI.

        libCZI reads only the sub-blocks that intersect ``region``, so a window costs
        the tiles it actually covers rather than every tile in the plane. That is the
        whole point of stitching this way rather than reading all tiles and pasting
        them together in Python.

        Parameters
        ----------
        source: CziSource
            Where the image lives and how to reopen it.
        read_dims: Dict[str, int]
            CZI-native dimension chars to absolute indices, pinning one plane.
        region: Tuple[int, int, int, int]
            (x, y, width, height) in the file's mosaic coordinate frame.
        out_shape: Tuple[int, ...]
            Shape to return. read_mosaic prepends a size-1 axis per pinned dimension,
            which is reshaped away here.
        dtype: np.dtype
            The dtype the caller declared to dask. A mismatch would hand dask
            silently wrong data, so it is checked rather than trusted.
        """
        with source.open() as czi:
            data = czi.read_mosaic(region=region, scale_factor=1.0, **read_dims)

        if data.dtype != dtype:
            raise TypeError(
                f"Mosaic region {region} read back as {data.dtype}, but the mosaic "
                f"was declared as {dtype}."
            )
        return data.reshape(out_shape)

    def _mosaic_plane_reads(
        self, czi: CziFile
    ) -> Tuple[List[str], List[int], List[int], BBox]:
        """
        Work out how to address one composited plane of the current scene.

        Returns the CZI-native chars of the dimensions that must be pinned for a
        mosaic read, their sizes, the absolute index each one starts at, and the
        bounding box of the current scene within the mosaic.
        """
        dims_shape = Reader._dims_shape_to_scene_dims_shape(
            dims_shape=czi.get_dims_shape(),
            scene_index=self.current_scene_index,
            consistent=czi.shape_is_consistent,
        )
        sizes = dict(zip(self.dims.order, self.dims.shape))
        spatial = (
            DimensionNames.MosaicTile,
            DimensionNames.SpatialY,
            DimensionNames.SpatialX,
            DimensionNames.Samples,
        )
        # libCZI composites a single plane at a time, so every dimension other than
        # the tile and spatial ones has to be pinned to one index per read.
        plane_dims = [d for d in self.dims.order if d not in spatial]
        czi_chars = [_BIOIO_TO_CZI_DIM.get(d, d) for d in plane_dims]
        plane_sizes = [sizes[d] for d in plane_dims]
        begins = [dims_shape[c][0] for c in czi_chars]
        bbox = czi.get_mosaic_scene_bounding_box(index=self.czi_scene_index)
        return czi_chars, plane_sizes, begins, bbox

    def _stitched_mosaic_dask(self) -> da.Array:
        """
        Build the stitched mosaic as a grid of lazily-read regions.

        Each chunk is one ``read_mosaic`` call over its own rectangle, so slicing a
        window out of the result reads only the tiles under that window. Chunking by
        tile keeps a small window down to a single read; callers that intend to pull
        the whole mosaic can trade that for fewer, larger reads with
        ``mosaic_chunk_size``, because tiles overlap and a tile-sized grid therefore
        straddles more sub-blocks than a coarser one.
        """
        with self._source.open() as czi:
            self._require_mosaic(czi)
            czi_chars, plane_sizes, begins, bbox = self._mosaic_plane_reads(czi)
            pixel_type = PIXEL_DICT.get(czi.pixel_type)
            if pixel_type is None:
                raise TypeError(
                    f"Unsupported or unlabeled pixel type: {czi.pixel_type!r}"
                )

        sizes = dict(zip(self.dims.order, self.dims.shape))
        n_samples = sizes.get(DimensionNames.Samples)
        chunk_h, chunk_w = self._mosaic_chunk_size or (
            sizes[DimensionNames.SpatialY],
            sizes[DimensionNames.SpatialX],
        )
        y_starts = list(range(0, bbox.h, chunk_h))
        x_starts = list(range(0, bbox.w, chunk_w))

        # One entry per (plane, chunk row, chunk column). The trailing singleton is
        # the Samples axis, which da.block concatenates rather than stacks.
        grid_shape: Tuple[int, ...] = tuple(plane_sizes) + (
            len(y_starts),
            len(x_starts),
        )
        if n_samples is not None:
            grid_shape += (1,)
        lazy_arrays: np.ndarray = np.ndarray(grid_shape, dtype=object)

        for index, _ in np.ndenumerate(lazy_arrays):
            plane_index = index[: len(plane_sizes)]
            y_start = y_starts[index[len(plane_sizes)]]
            x_start = x_starts[index[len(plane_sizes) + 1]]
            height = min(chunk_h, bbox.h - y_start)
            width = min(chunk_w, bbox.w - x_start)
            out_shape: Tuple[int, ...] = (height, width)
            if n_samples is not None:
                out_shape += (n_samples,)

            lazy_arrays[index] = da.from_delayed(
                delayed(Reader._read_mosaic_region)(
                    source=self._source,
                    read_dims={
                        char: begin + position
                        for char, begin, position in zip(czi_chars, begins, plane_index)
                    },
                    # The scene's origin within the mosaic is not the origin of the
                    # mosaic itself; plate files put scenes at large offsets.
                    region=(bbox.x + x_start, bbox.y + y_start, width, height),
                    out_shape=out_shape,
                    dtype=pixel_type,
                ),
                shape=out_shape,
                dtype=pixel_type,
            )

        return da.block(lazy_arrays.tolist())

    def _stitched_mosaic_numpy(self) -> np.ndarray:
        """
        Read the whole stitched mosaic into memory, one composite per plane.

        Deliberately not ``_stitched_mosaic_dask().compute()``: tiles overlap, so a
        chunk grid covering the entire mosaic touches noticeably more sub-blocks than
        one read per plane does. The chunked path exists to make windows cheap, not
        whole reads.
        """
        with self._source.open() as czi:
            self._require_mosaic(czi)
            czi_chars, plane_sizes, begins, bbox = self._mosaic_plane_reads(czi)
            pixel_type = PIXEL_DICT.get(czi.pixel_type)
            if pixel_type is None:
                raise TypeError(
                    f"Unsupported or unlabeled pixel type: {czi.pixel_type!r}"
                )

            sizes = dict(zip(self.dims.order, self.dims.shape))
            n_samples = sizes.get(DimensionNames.Samples)
            plane_shape: Tuple[int, ...] = (bbox.h, bbox.w)
            if n_samples is not None:
                plane_shape += (n_samples,)

            out = np.empty(tuple(plane_sizes) + plane_shape, dtype=pixel_type)
            for plane_index in itertools.product(*(range(s) for s in plane_sizes)):
                read_dims = {
                    char: begin + position
                    for char, begin, position in zip(czi_chars, begins, plane_index)
                }
                data = czi.read_mosaic(
                    region=(bbox.x, bbox.y, bbox.w, bbox.h),
                    scale_factor=1.0,
                    **read_dims,
                )
                out[plane_index] = data.reshape(plane_shape)

        return out

    @staticmethod
    def _require_mosaic(czi: CziFile) -> None:
        """
        Raise unless this image actually has tiles to stitch.
        """
        if not czi.is_mosaic():
            raise exceptions.InvalidDimensionOrderingError(
                "Cannot create stitched mosaic image for array without tiles "
                "available."
            )

    def _get_stitched_dask_mosaic(self) -> xr.DataArray:
        return self._construct_mosaic_xarray(self._stitched_mosaic_dask())

    def _get_stitched_mosaic(self) -> xr.DataArray:
        return self._construct_mosaic_xarray(self._stitched_mosaic_numpy())

    @property
    def czi_scene_index(self) -> int:
        """
        Returns the plate-level CZI scene index for the current scene.

        For split CZI files (one plate position per file), this differs from
        current_scene_index (which is always 0) because the embedded XML metadata
        still uses the original plate-wide scene indices.
        """
        if self._czi_scene_index is None:
            with self._source.open() as czi:
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

        with self._source.open() as czi:

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

        with self._source.open() as czi:

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

        with self._source.open() as czi:
            return acquisition_times(
                czi=czi,
                current_scene=self.czi_scene_index,
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

    @property
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
            with self._source.open() as czi:
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
