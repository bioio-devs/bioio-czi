#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pathlib
from typing import Any, Optional, Tuple
from unittest import mock

import numpy as np
import pytest
from bioio_base import exceptions, types
from fsspec.implementations.local import LocalFileSystem

from bioio_czi import Reader, handle_pool, remote
from bioio_czi.aicspylibczi_reader import reader as aicspylibczi_reader
from bioio_czi.aicspylibczi_reader.reader import CziSource, remote_reads_available

from .conftest import LOCAL_RESOURCES_DIR

READER_NAME = "test-reader"

HTTPS_URL = "https://example.com/data/image.czi"
S3_URI = "s3://bucket/prefix/image.czi"


class FakeObjectStore:
    """
    Stand-in for an object-store filesystem, so these tests need no credentials.

    Only the two methods :mod:`bioio_czi.remote` uses are implemented.
    """

    def __init__(
        self,
        protocol: str = "s3",
        signed: Optional[str] = None,
        sign_error: Optional[Exception] = None,
    ) -> None:
        self.protocol = protocol
        self._signed = signed
        self._sign_error = sign_error
        self.sign_calls: list[Tuple[str, int]] = []

    def unstrip_protocol(self, path: str) -> str:
        return f"{self.protocol}://{path}"

    def sign(self, path: str, expiration: int = 100, **kwargs: Any) -> str:
        self.sign_calls.append((path, expiration))
        if self._sign_error is not None:
            raise self._sign_error
        assert self._signed is not None
        return self._signed


@pytest.mark.parametrize(
    "image, expected_scheme, expected_http, expected_remote",
    [
        (pathlib.Path("/images/a.czi"), "", False, False),
        ("/images/a.czi", "", False, False),
        # A single letter before the colon is a Windows drive, not a protocol.
        ("C:\\images\\a.czi", "", False, False),
        ("file:///images/a.czi", "file", False, False),
        (HTTPS_URL, "https", True, True),
        ("http://example.com/a.czi", "http", True, True),
        # Schemes are matched case-insensitively.
        ("HTTPS://example.com/a.czi", "https", True, True),
        (S3_URI, "s3", False, True),
        ("gs://bucket/a.czi", "gs", False, True),
    ],
)
def test_uri_classification(
    image: types.PathLike,
    expected_scheme: str,
    expected_http: bool,
    expected_remote: bool,
) -> None:
    assert remote.uri_scheme(image) == expected_scheme
    assert remote.is_http_url(image) is expected_http
    assert remote.is_remote(image) is expected_remote
    assert remote.is_local(image) is not expected_remote


def test_resolve_url_passes_http_through() -> None:
    # An http(s) URL is what libCZI wants already, so it is handed over untouched
    # rather than round-tripped through a filesystem.
    assert remote.resolve_url(HTTPS_URL, reader_name=READER_NAME) == HTTPS_URL


def test_resolve_url_rejects_local_path() -> None:
    with pytest.raises(ValueError):
        remote.resolve_url("/images/a.czi", reader_name=READER_NAME)


def test_resolve_url_signs_object_store(monkeypatch: pytest.MonkeyPatch) -> None:
    fs = FakeObjectStore(signed=f"{HTTPS_URL}?signature=abc")
    monkeypatch.setattr(
        remote.fsspec.core,
        "url_to_fs",
        lambda uri, **kwargs: (fs, "bucket/prefix/image.czi"),
    )

    url = remote.resolve_url(S3_URI, reader_name=READER_NAME, expiration=42)

    assert url == f"{HTTPS_URL}?signature=abc"
    assert fs.sign_calls == [("bucket/prefix/image.czi", 42)]


def test_resolve_url_reports_missing_filesystem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def raise_import_error(uri: str, **kwargs: Any) -> Any:
        raise ImportError("Install s3fs to access S3")

    monkeypatch.setattr(remote.fsspec.core, "url_to_fs", raise_import_error)

    with pytest.raises(exceptions.UnsupportedFileFormatError, match="s3fs"):
        remote.resolve_url(S3_URI, reader_name=READER_NAME)


def test_sign_url_reports_unsignable_filesystem() -> None:
    fs = FakeObjectStore(sign_error=NotImplementedError())

    with pytest.raises(exceptions.UnsupportedFileFormatError, match="presigned"):
        remote.sign_url(fs, "bucket/key", uri=S3_URI, reader_name=READER_NAME)


def test_sign_url_rejects_non_http_signature() -> None:
    # A filesystem that "signs" to something libCZI cannot fetch is a bug we want
    # reported at the reader, not a confusing failure inside libCZI.
    fs = FakeObjectStore(signed="s3://bucket/prefix/image.czi?signature=abc")

    with pytest.raises(exceptions.UnsupportedFileFormatError, match="not an http"):
        remote.sign_url(fs, "bucket/key", uri=S3_URI, reader_name=READER_NAME)


@pytest.mark.parametrize(
    "path, expected_uri, expected_remote",
    [
        (HTTPS_URL, HTTPS_URL, True),
        # fsspec hands http(s) paths back with their scheme intact; putting the
        # protocol back on would double it up ("https://http://...").
        ("http://example.com/a.czi", "http://example.com/a.czi", True),
        (
            "https://example.com/a.czi?versionId=xyz",
            "https://example.com/a.czi?versionId=xyz",
            True,
        ),
    ],
)
def test_czi_source_from_http_filesystem(
    path: str, expected_uri: str, expected_remote: bool
) -> None:
    source = CziSource.from_fs(FakeObjectStore(protocol="https"), path)

    assert source.uri == expected_uri
    assert source.is_remote is expected_remote


def test_czi_source_from_object_store_filesystem() -> None:
    # Object-store paths arrive protocol-stripped, so the protocol is put back on
    # in order to presign later.
    source = CziSource.from_fs(FakeObjectStore(), "bucket/prefix/image.czi")

    assert source.uri == S3_URI
    assert source.is_remote is True


# The tests below serve real CZIs over http from localhost, which only exercises
# the aicspylibczi backend: pylibCZIrw validates URLs with the ``validators``
# package, which rejects hostnames with no TLD ("http://localhost:8000/a.czi"),
# and its curl reader stalls on a loopback IP without ever opening a connection.
# Remote reading in pylibczirw mode is covered against a real server in
# test_pylibczirw_reader.py instead.
requires_remote_reads = pytest.mark.skipif(
    not remote_reads_available(),
    reason="This aicspylibczi build was compiled without libCZI's curl stream",
)


@requires_remote_reads
def test_reader_over_http_matches_local(local_http_server: str) -> None:
    # The same bytes served over http must produce the same image as reading off
    # disk, metadata included.
    filename = "s_1_t_1_c_1_z_1.czi"
    local = Reader(LOCAL_RESOURCES_DIR / filename, use_aicspylibczi=True)
    over_http = Reader(f"{local_http_server}/{filename}", use_aicspylibczi=True)

    assert over_http.scenes == local.scenes
    assert over_http.dims.order == local.dims.order
    assert over_http.shape == local.shape
    assert over_http.channel_names == local.channel_names
    assert over_http.physical_pixel_sizes == local.physical_pixel_sizes
    np.testing.assert_array_equal(over_http.data, local.data)
    # The delayed path locates the remote image from inside the dask graph, so it
    # reaches it independently of the eager path above.
    np.testing.assert_array_equal(over_http.dask_data.compute(), local.data)


@requires_remote_reads
def test_reader_reuses_remote_handles(local_http_server: str) -> None:
    # Opening a remote CZI refetches its header, metadata and sub-block directory
    # before any pixels are read, so repeated reads have to share a handle rather
    # than paying that every time.
    handle_pool.clear_pools()

    opens = 0
    original = aicspylibczi_reader.CziSource._open_remote

    def counting_open(self: aicspylibczi_reader.CziSource) -> Any:
        nonlocal opens
        opens += 1
        return original(self)

    with mock.patch.object(
        aicspylibczi_reader.CziSource, "_open_remote", counting_open
    ):
        reader = Reader(
            f"{local_http_server}/s_3_t_1_c_3_z_5.czi", use_aicspylibczi=True
        )
        for channel in range(3):
            reader.get_image_data("YX", C=channel, Z=0, Y=slice(0, 32), X=slice(0, 32))

    # Construction alone inspects the image several times, and each read opens it
    # again; serially, one handle serves all of it.
    assert opens == 1
    handle_pool.clear_pools()


@requires_remote_reads
def test_local_reads_are_not_pooled() -> None:
    # A pooled local handle would hold the file open for the life of the process for
    # no gain: reopening a local CZI is immeasurably cheap next to reading one.
    handle_pool.clear_pools()
    reader = Reader(LOCAL_RESOURCES_DIR / "s_3_t_1_c_3_z_5.czi", use_aicspylibczi=True)
    reader.get_image_data("YX", C=0, Z=0)

    assert not handle_pool._registry


@requires_remote_reads
def test_reader_over_http_reads_sub_region(local_http_server: str) -> None:
    # Sub-region reads are the point of reading over range requests, so check one
    # matches the equivalent local read rather than only that it has the right shape.
    filename = "s_3_t_1_c_3_z_5.czi"
    local = Reader(LOCAL_RESOURCES_DIR / filename, use_aicspylibczi=True)
    over_http = Reader(f"{local_http_server}/{filename}", use_aicspylibczi=True)
    over_http.set_scene(1)
    local.set_scene(1)

    np.testing.assert_array_equal(
        over_http.get_image_data("YX", C=1, Z=2, Y=slice(0, 32), X=slice(0, 32)),
        local.get_image_data("YX", C=1, Z=2, Y=slice(0, 32), X=slice(0, 32)),
    )


@requires_remote_reads
def test_reader_reports_missing_remote_image(local_http_server: str) -> None:
    with pytest.raises((FileNotFoundError, exceptions.UnsupportedFileFormatError)):
        Reader(f"{local_http_server}/no-such-image.czi", use_aicspylibczi=True)


@requires_remote_reads
def test_reader_rejects_remote_non_czi(local_http_server: str) -> None:
    with pytest.raises(exceptions.UnsupportedFileFormatError):
        Reader(f"{local_http_server}/s_1_t_1_c_1_z_1.ome.tiff", use_aicspylibczi=True)


@pytest.mark.parametrize("use_aicspylibczi", [False, True])
def test_reader_reads_object_store_uri(use_aicspylibczi: bool) -> None:
    # The full presign path against a real object store: fsspec turns the s3:// URI
    # into an https URL and libCZI range-reads it. The bucket is public, so this
    # reads anonymously rather than needing credentials in CI.
    pytest.importorskip("s3fs", reason="s3fs is needed to presign 's3://' URIs")
    if use_aicspylibczi and not remote_reads_available():
        pytest.skip("This aicspylibczi build was compiled without libCZI's curl stream")

    uri = (
        "s3://allencell/aics/hipsc_12x_overview_image_dataset/"
        "stitchedwelloverviewimagepath/05080558_3500003720_10X_20191220_D3.czi"
    )
    reader = Reader(uri, use_aicspylibczi=use_aicspylibczi, fs_kwargs={"anon": True})

    assert reader.shape[-2:] == (5684, 5925)
    window = reader.get_image_data("YX", Y=slice(0, 32), X=slice(0, 32))
    assert window.shape == (32, 32)


def test_czi_source_from_local_filesystem() -> None:
    path = str(LOCAL_RESOURCES_DIR / "s_1_t_1_c_1_z_1.czi")
    source = CziSource.from_fs(LocalFileSystem(), path)

    # Local paths stay as they are rather than becoming file:// URIs.
    assert source.uri == path
    assert source.is_remote is False
    with source.open() as czi:
        assert czi.dims == "BCYX"
