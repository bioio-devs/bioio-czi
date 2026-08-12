"""
Reuse open CZI handles instead of reopening the file for every read.

Neither backend holds a handle between calls, so each read reopens the image. Locally
that costs well under a millisecond. Remotely it means re-fetching the file header,
metadata and sub-block directory before a single pixel is read -- measured at 0.6-0.9s
against S3 -- and it is paid again by every ``get_image_data`` call and every chunk of
a dask graph.

Pools live here, in a process-global registry keyed by how to reach the image, rather
than on the readers. A reader is pickled into dask graphs; an open handle is not
picklable and a handle carrying a presigned URL should not outlive its signature
anyway. Keeping pools out of the readers means a graph still ships only "how to
reopen", while each worker process builds and reuses its own handles.

Handles are pooled rather than shared through one slot because libCZI's curl stream
serializes requests per handle: several dask threads reading at once want a connection
each, not a queue behind one.
"""

import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Callable, Hashable, Iterator, List, Optional, Tuple

# How many idle handles one image keeps. Enough to cover a normal dask thread pool
# without holding open a connection per chunk.
DEFAULT_MAX_HANDLES = 8

# How many images keep pools at once. Bounded so that walking a plate directory does
# not accumulate open handles for every file it touches.
MAX_POOLED_IMAGES = 16


class HandlePool:
    """
    A small set of interchangeable open handles for one image.

    Parameters
    ----------
    factory: Callable[[], Any]
        Opens a new handle. Called outside the lock, so a slow open does not block
        other threads from returning theirs.
    max_size: int
        How many idle handles to keep. Extra handles are closed on release rather
        than pooled.
        Default: DEFAULT_MAX_HANDLES
    ttl: Optional[float]
        Seconds after which a handle is discarded rather than reused. Used for
        presigned URLs, which stop working once the signature expires.
        Default: None (handles never go stale)
    closer: Optional[Callable[[Any], None]]
        Releases a handle. Needed for readers that must be closed explicitly.
        Default: None
    """

    def __init__(
        self,
        factory: Callable[[], Any],
        *,
        max_size: int = DEFAULT_MAX_HANDLES,
        ttl: Optional[float] = None,
        closer: Optional[Callable[[Any], None]] = None,
    ) -> None:
        self._factory = factory
        self._max_size = max_size
        self._ttl = ttl
        self._closer = closer
        self._idle: List[Tuple[Any, float]] = []
        self._lock = threading.Lock()

    def _is_fresh(self, opened_at: float) -> bool:
        return self._ttl is None or (time.monotonic() - opened_at) < self._ttl

    @contextmanager
    def acquire(self) -> Iterator[Any]:
        """
        Take a handle for the duration of the block, then return it to the pool.

        A handle whose block raised is discarded rather than pooled: an error out of
        a read means the handle itself is suspect, most often an expired signature or
        a dropped connection.
        """
        handle = None
        opened_at = 0.0
        with self._lock:
            while self._idle:
                candidate, candidate_opened_at = self._idle.pop()
                if self._is_fresh(candidate_opened_at):
                    handle, opened_at = candidate, candidate_opened_at
                    break
                self._discard(candidate)

        if handle is None:
            opened_at = time.monotonic()
            handle = self._factory()

        failed = False
        try:
            yield handle
        except Exception:
            failed = True
            raise
        finally:
            with self._lock:
                stale = failed or not self._is_fresh(opened_at)
                if stale or len(self._idle) >= self._max_size:
                    self._discard(handle)
                else:
                    self._idle.append((handle, opened_at))

    def _discard(self, handle: Any) -> None:
        if self._closer is None:
            return
        try:
            self._closer(handle)
        except Exception:
            # A handle is only discarded when it is already being thrown away, so a
            # failure to close it has nothing left to affect.
            pass

    def close(self) -> None:
        """
        Close every idle handle. Handles currently in use are closed on release.
        """
        with self._lock:
            for handle, _ in self._idle:
                self._discard(handle)
            self._idle.clear()


_registry: "OrderedDict[Hashable, HandlePool]" = OrderedDict()
_registry_lock = threading.Lock()


def get_pool(
    key: Hashable,
    factory: Callable[[], Any],
    *,
    max_size: int = DEFAULT_MAX_HANDLES,
    ttl: Optional[float] = None,
    closer: Optional[Callable[[Any], None]] = None,
) -> HandlePool:
    """
    Return the pool for ``key``, creating it if this process has not seen it.

    ``key`` must capture everything that changes what a handle opens -- the location
    and any options used to reach it -- because handles are shared by every reader
    that asks for the same key.
    """
    with _registry_lock:
        pool = _registry.get(key)
        if pool is not None:
            _registry.move_to_end(key)
        if pool is None:
            while len(_registry) >= MAX_POOLED_IMAGES:
                # Least recently used first.
                _, evicted = _registry.popitem(last=False)
                evicted.close()
            pool = HandlePool(factory, max_size=max_size, ttl=ttl, closer=closer)
            _registry[key] = pool
        return pool


def clear_pools() -> None:
    """
    Close and forget every pooled handle. Intended for tests and for callers that
    want to release file handles and connections deterministically.
    """
    with _registry_lock:
        for pool in _registry.values():
            pool.close()
        _registry.clear()
