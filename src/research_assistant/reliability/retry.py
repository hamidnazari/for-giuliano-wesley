"""Retry with exponential backoff, and call timeouts."""

import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout


class RetryableError(Exception):
    """A transient failure worth retrying (rate limit, network blip, 5xx)."""


def call_with_timeout[T](fn: Callable[[], T], seconds: float) -> T:
    """Run fn and raise TimeoutError if it takes longer.

    The worker thread cannot be killed, so a timed-out call may finish in the background;
    callers must treat the result as lost.
    """
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        return pool.submit(fn).result(timeout=seconds)
    except FutureTimeout:
        raise TimeoutError(f"call exceeded {seconds:g}s") from None
    finally:
        pool.shutdown(wait=False)


def call_with_retry[T](
    fn: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call fn, retrying RetryableError and TimeoutError with exponential backoff."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    for attempt in range(attempts):
        try:
            return fn()
        except (RetryableError, TimeoutError):
            if attempt == attempts - 1:
                raise
            sleep(min(max_delay, base_delay * 2**attempt))
    raise AssertionError("unreachable")
