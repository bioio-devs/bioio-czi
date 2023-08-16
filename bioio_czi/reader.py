#!/usr/bin/env python
# -*- coding: utf-8 -*-

from typing import Any, Optional, Tuple

from bioio_base.dimensions import Dimensions
from bioio_base.reader import Reader as BaseReader

import xarray as xr
from fsspec.spec import AbstractFileSystem

###############################################################################


class Reader(BaseReader):
    """
    The main class of each reader plugin. This class is subclass
    of the abstract class reader (BaseReader) in bioio-base.

    Parameters
    ----------
    image: Any
        Some type of object to read and follow the Reader specification.
    fs_kwargs: Dict[str, Any]
        Any specific keyword arguments to pass down to the fsspec created filesystem.
        Default: {}

    Notes
    -----
    It is up to the implementer of the Reader to decide which types they would like to
    accept (certain readers may not support buffers for example).

    """
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

    # Required Methods

    def __init__(image: Any, **kwargs: Any):
        """
        Store / cache certain parameters required for later reading.

        Try not to read the image into memory here.
        """
        raise NotImplementedError()

    @staticmethod
    def _is_supported_image(fs: "AbstractFileSystem", path: str, **kwargs: Any) -> bool:
        """
        Perform a check to determine if the object(s) or path(s) provided as
        parameters are supported by this reader.
        """
        raise NotImplementedError()

    @property
    def scenes(self) -> Tuple[str, ...]:
        """
        Return the list of available scenes for the file using the
        cached parameters stored to the object in the __init__.
        """
        raise NotImplementedError()

    def _read_delayed(self) -> "xr.DataArray":
        """
        Return an xarray DataArray filled with a delayed dask array, coordinate planes,
        and any metadata stored in the attrs.

        Metadata should be labelled with one of the bioio-base constants.
        """
        raise NotImplementedError()

    def _read_immediate(self) -> "xr.DataArray":
        """
        Return an xarray DataArray filled with an in-memory numpy ndarray,
        coordinate planes, and any metadata stored in the attrs.

        Metadata should be labelled with one of the bioio-base constants.
        """
        raise NotImplementedError()

    # Optional Methods

    def _get_stitched_dask_mosaic(self) -> "xr.DataArray":
        """
        If your file returns an `M` dimension for "Mosiac Tile",
        this function should stitch and return the stitched data as
        an xarray DataArray both operating against the original delayed array
        and returning a delayed array.
        """
        return super()._get_stitched_dask_mosaic()

    def _get_stitched_mosaic(self) -> "xr.DataArray":
        """
        If your file returns an `M` dimension for "Mosiac Tile",
        this function should stitch and return the stitched data as
        an xarray DataArray both operating against the original in-memory array
        and returning a in-memory array.
        """
        return super()._get_stitched_mosaic()
