#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pytest
from bioio_base import exceptions

from bioio_czi import Reader
from bioio_czi.aicspylibczi_reader.reader import remote_reads_available

from .conftest import LOCAL_RESOURCES_DIR

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
