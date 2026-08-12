import functools
import http.server
import io
import os
import pathlib
import re
import threading
from typing import Any, BinaryIO, Iterator, Optional

import pytest

LOCAL_RESOURCES_DIR = pathlib.Path(__file__).parent / "resources"


class _RangeRequestHandler(http.server.SimpleHTTPRequestHandler):
    """
    A static file handler that honors ``Range`` requests.

    libCZI reads a remote CZI by asking for the byte ranges holding the sub-blocks
    it needs, so the stock handler -- which only ever sends whole files -- cannot
    stand in for a real image server.
    """

    protocol_version = "HTTP/1.1"

    def send_head(self) -> Optional[BinaryIO]:
        path = self.translate_path(self.path)
        try:
            handle = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        size = os.fstat(handle.fileno()).st_size
        range_header = self.headers.get("Range")
        if range_header is None:
            body: BinaryIO = handle
            length = size
            status = 200
        else:
            match = re.fullmatch(r"bytes=(\d+)-(\d*)", range_header.strip())
            if match is None:
                handle.close()
                self.send_error(416, f"Unsupported range: {range_header}")
                return None
            start = int(match.group(1))
            end = min(int(match.group(2)) if match.group(2) else size - 1, size - 1)
            if start > end:
                handle.close()
                self.send_error(416, f"Range not satisfiable: {range_header}")
                return None
            with handle:
                handle.seek(start)
                body = io.BytesIO(handle.read(end - start + 1))
            length = end - start + 1
            status = 206

        self.send_response(status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(length))
        self.send_header("Accept-Ranges", "bytes")
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        return body

    def log_message(self, format: str, *args: Any) -> None:
        # Keep the test output readable; every read makes several requests.
        pass


@pytest.fixture(scope="session")
def local_http_server() -> Iterator[str]:
    """
    Serve the test resources over http, with range support, on localhost.

    Yields the base URL, e.g. "http://127.0.0.1:54321", so tests can read the same
    resource files over the network path that a remote image takes.
    """
    handler = functools.partial(
        _RangeRequestHandler, directory=str(LOCAL_RESOURCES_DIR)
    )
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
