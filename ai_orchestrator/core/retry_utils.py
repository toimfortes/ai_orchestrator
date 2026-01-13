"""Tenacity-based retry utilities for CLI invocations.

Provides industry-standard retry logic with exponential backoff and jitter
to prevent thundering herd problems in multi-agent systems.
"""

from __future__ import annotations

import logging
from typing import Callable, TypeVar

from tenacity import (
    RetryCallState,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RetryableError(Exception):
    """Exception that should trigger retry logic."""

    def __init__(self, message: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


class TimeoutRetryError(RetryableError):
    """Timeout error that should be retried."""

    pass


class RateLimitRetryError(RetryableError):
    """Rate limit error that should be retried with longer backoff."""

    pass


class AuthError(RetryableError):
    """Authentication error - should NOT be retried."""

    def __init__(self, message: str) -> None:
        super().__init__(message, retryable=False)


def log_retry_attempt(retry_state: RetryCallState) -> None:
    """Log retry attempts for debugging."""
    if retry_state.attempt_number > 1:
        logger.warning(
            "Retry attempt %d after %.2fs (exception: %s)",
            retry_state.attempt_number,
            retry_state.seconds_since_start,
            retry_state.outcome.exception() if retry_state.outcome else "unknown",
        )


def create_retry_decorator(
    max_attempts: int = 3,
    min_wait: float = 1.0,
    max_wait: float = 60.0,
    retry_exceptions: tuple[type[Exception], ...] = (
        TimeoutRetryError,
        RateLimitRetryError,
        ConnectionError,
        TimeoutError,
    ),
) -> Callable:
    """
    Create a tenacity retry decorator with exponential backoff and jitter.

    This follows best practices from Gemini deep research:
    - Uses wait_random_exponential for jitter to prevent thundering herd
    - Configurable retry exceptions for selective retry
    - Logging for observability

    Args:
        max_attempts: Maximum number of retry attempts.
        min_wait: Minimum wait time in seconds.
        max_wait: Maximum wait time cap in seconds.
        retry_exceptions: Tuple of exception types to retry on.

    Returns:
        A tenacity retry decorator.

    Example:
        @create_retry_decorator(max_attempts=3)
        async def call_api():
            ...
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_random_exponential(multiplier=min_wait, max=max_wait),
        retry=retry_if_exception_type(retry_exceptions),
        before_sleep=log_retry_attempt,
        reraise=True,
    )


# Pre-configured decorators for common use cases
retry_with_backoff = create_retry_decorator(
    max_attempts=3,
    min_wait=1.0,
    max_wait=60.0,
)

retry_rate_limited = create_retry_decorator(
    max_attempts=5,
    min_wait=2.0,
    max_wait=120.0,  # Longer max for rate limits
    retry_exceptions=(RateLimitRetryError,),
)

retry_timeout = create_retry_decorator(
    max_attempts=2,
    min_wait=1.0,
    max_wait=30.0,
    retry_exceptions=(TimeoutRetryError, TimeoutError),
)
