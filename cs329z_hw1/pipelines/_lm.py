"""Shared helper: one model call with retries for transient provider errors."""

from __future__ import annotations

import time

from cs329z_hw1.llm import BudgetExceeded
from cs329z_hw1.types import LMCallable, Message

MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 2.0


def call_lm(lm: LMCallable, messages: list[Message]) -> str:
    """Call ``lm``; retry up to MAX_ATTEMPTS times on errors.

    Under concurrency the provider occasionally fails a call (rate limit,
    timeout) even after the wrapper's own retries. A failed call is not
    cached, so trying again is safe. BudgetExceeded is final and re-raised,
    as is the last error once the attempts are used up.
    """
    last_error: Exception | None = None
    for attempt in range(MAX_ATTEMPTS):
        if attempt:
            time.sleep(BACKOFF_SECONDS * 2 ** (attempt - 1))  # 2s, 4s: let a rate limit clear
        try:
            reply = lm(messages)
        except BudgetExceeded:
            raise
        except Exception as exc:
            last_error = exc
            continue
        return reply if isinstance(reply, str) else str(reply)
    assert last_error is not None
    raise last_error
