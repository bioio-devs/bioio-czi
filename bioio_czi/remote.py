"""
Locate CZIs that live somewhere other than the local filesystem.

Both backends read remote CZIs the same way: libCZI's curl-based stream pulls byte
ranges over http(s) directly, so only the sub-blocks actually needed cross the
network and no bytes flow through Python. pylibCZIrw exposes this as
``ReaderFileInputTypes.Curl``; aicspylibczi accepts an http/https URL in place of a
filename.

Object stores addressed by their own protocol -- ``s3://``, ``gs://``, ``az://`` --
are supported by asking the matching fsspec filesystem to presign the object into an
https URL, which is then handed to libCZI like any other URL. This keeps the fast
range-request read path rather than downloading the whole file, and means
credentials stay with fsspec.
"""

import logging
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.parse import urlparse

import fsspec
from bioio_base import exceptions, types
from fsspec.spec import AbstractFileSystem

log = logging.getLogger(__name__)

# Schemes libCZI's curl stream reads natively.
HTTP_SCHEMES = frozenset({"http", "https"})

# Schemes that address the local filesystem rather than a network location.
LOCAL_SCHEMES = frozenset({"", "file", "local"})

# Object-store schemes we expect to be able to presign. This list is only used to
# make error messages actionable: any non-local protocol whose fsspec filesystem
# implements ``sign`` will work, whether or not it is named here.
KNOWN_OBJECT_STORE_SCHEMES = ("s3", "gs", "gcs", "az", "abfs", "abfss", "adl")

# How long a generated presigned URL stays valid. libCZI issues range requests for
# the whole life of a reader -- and, in aicspylibczi mode, for the life of any dask
# graph built from it -- so this needs to outlast a full read session rather than a
# single request.
DEFAULT_URL_EXPIRATION_SECONDS = 3600


def uri_scheme(image: types.PathLike) -> str:
    """
    Return the URI scheme of ``image``, or "" if it names a local path.

    Parameters
    ----------
    image: types.PathLike
        A path or URI, as given by the caller.

    Returns
    -------
    scheme: str
        The lowercased scheme, e.g. "s3" for "s3://bucket/key". "" for local paths.
    """
    if isinstance(image, Path):
        return ""
    scheme = urlparse(str(image)).scheme.lower()
    # A single-character scheme is a Windows drive letter ("C:\\images\\a.czi"),
    # not a protocol.
    return "" if len(scheme) < 2 else scheme


def is_http_url(image: types.PathLike) -> bool:
    """
    Whether ``image`` is an http/https URL, which libCZI can read as-is.
    """
    return uri_scheme(image) in HTTP_SCHEMES


def is_local(image: types.PathLike) -> bool:
    """
    Whether ``image`` names a file on the local filesystem.
    """
    return uri_scheme(image) in LOCAL_SCHEMES


def is_remote(image: types.PathLike) -> bool:
    """
    Whether ``image`` must be reached over the network.

    Returns True both for http(s) URLs and for object-store protocols such as
    ``s3://``, which are presigned into http(s) URLs by :func:`resolve_url`.
    """
    return not is_local(image)


def resolve_url(
    image: types.PathLike,
    *,
    reader_name: str,
    expiration: int = DEFAULT_URL_EXPIRATION_SECONDS,
    fs_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Return an http(s) URL that libCZI's curl stream can read ``image`` from.

    Parameters
    ----------
    image: types.PathLike
        A remote URI. http(s) URLs are returned unchanged; any other protocol is
        presigned by its fsspec filesystem.
    reader_name: str
        Name of the calling reader, used in error messages.
    expiration: int
        Seconds a generated presigned URL stays valid.
        Default: DEFAULT_URL_EXPIRATION_SECONDS
    fs_kwargs: Optional[Dict[str, Any]]
        Keyword arguments for the fsspec filesystem used to presign.
        Default: None

    Returns
    -------
    url: str
        An http or https URL.

    Raises
    ------
    exceptions.UnsupportedFileFormatError
        The protocol has no fsspec implementation installed, or that
        implementation cannot presign.
    ValueError
        ``image`` is a local path, so there is no URL to resolve.
    """
    uri = str(image)
    if is_http_url(uri):
        return uri
    if is_local(uri):
        raise ValueError(f"{uri!r} is a local path, so it has no remote URL.")

    scheme = uri_scheme(uri)
    try:
        fs, key = fsspec.core.url_to_fs(uri, **(fs_kwargs or {}))
    except ImportError as exc:
        raise exceptions.UnsupportedFileFormatError(
            reader_name,
            uri,
            f"Reading '{scheme}://' paths needs the fsspec filesystem for that "
            f"protocol to be installed: {exc}",
        )
    return sign_url(fs, key, uri=uri, reader_name=reader_name, expiration=expiration)


def sign_url(
    fs: AbstractFileSystem,
    path: str,
    *,
    uri: str,
    reader_name: str,
    expiration: int = DEFAULT_URL_EXPIRATION_SECONDS,
) -> str:
    """
    Presign ``path`` on an already-constructed filesystem into an https URL.

    Callers that already hold a filesystem should prefer this over
    :func:`resolve_url` so the filesystem (and its credentials) are not rebuilt.

    Parameters
    ----------
    fs: AbstractFileSystem
        The filesystem holding ``path``.
    path: str
        The protocol-stripped path within ``fs``, e.g. "bucket/key".
    uri: str
        The original URI including its protocol, used in error messages.
    reader_name: str
        Name of the calling reader, used in error messages.
    expiration: int
        Seconds the presigned URL stays valid.
        Default: DEFAULT_URL_EXPIRATION_SECONDS

    Returns
    -------
    url: str
        An http or https URL.

    Raises
    ------
    exceptions.UnsupportedFileFormatError
        ``fs`` cannot presign, or signed to something libCZI cannot read.
    """
    scheme = uri_scheme(uri)
    try:
        url = fs.sign(path, expiration=expiration)
    except NotImplementedError:
        raise exceptions.UnsupportedFileFormatError(
            reader_name,
            uri,
            f"The fsspec filesystem for '{scheme}://' cannot generate presigned "
            "URLs, which is how this reader hands remote files to libCZI. Pass an "
            "http/https URL instead, or use a protocol that supports signing "
            f"({', '.join(KNOWN_OBJECT_STORE_SCHEMES)}).",
        )

    if not is_http_url(url):
        raise exceptions.UnsupportedFileFormatError(
            reader_name,
            uri,
            f"The fsspec filesystem for '{scheme}://' signed to {url!r}, which is "
            "not an http/https URL and so cannot be read by libCZI.",
        )

    log.debug("Presigned %s for reading over http", uri)
    return url
