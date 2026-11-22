"""Problem 11 (compaction): helpers for keeping every model call within
``config.context_budget``. The policy that uses them is in ``session.py``.

The transcript is a log and is never shortened. Compaction only changes the
messages sent to the model, in these steps, cheapest first. Each runs only
while the conversation is still over the budget:

1. Stub old tool results. Tool results are the bulk of a long session and the
   least useful part of it once the model has read them and acted. A stub keeps
   the tool name, the status, the size and the first few words, so the model
   still sees that the step happened. The newest result is never stubbed: the
   model reads it in the call that follows it.
2. Summarize everything before the current request with ``aux_lm`` into one
   message. The prompt asks for what the user said they require first
   (budget, place, names, numbers), then what was done and what is pending.
3. Inside one very long request, drop its oldest steps (never the user's
   message), and as a last resort cut the newest result to fit.

Two things are never compacted away: the user's message being worked on, and
the newest tool result (cut to size only if it alone cannot fit).
"""

from __future__ import annotations

from typing import Any

from cs329z_hw1.llm import BudgetExceeded, OutputTruncated
from cs329z_hw1.tokens import count_tokens

STUB_MIN_TOKENS = 60  # a result shorter than this is cheaper than its stub
STUB_PREVIEW_CHARS = 80
SUMMARY_MAX_TOKENS = 700  # the summary itself must stay small, or it eats the budget
EVENT_CHARS = 1500  # per event, in what the summarizer reads
RESULT_CHARS = 400  # tool results are skimmed: the answer they led to is in a later message

SUMMARY_HEADER = "[Summary of the earlier conversation, which has left your context]\n"

SUMMARY_SYSTEM = """You compress the earlier part of a conversation between a user and an AI assistant so the assistant can continue it without the original messages. Write the summary as plain text, at most 250 words.

Start with a section "USER REQUIREMENTS" that lists, as bullets and copied exactly, everything the user said that must still hold: constraints, limits, budgets, amounts, places, dates, names, ids, preferences and corrections. Never drop or round a number or a name. If the user changed their mind, keep only the latest version.
Then a section "DONE" with what was asked and what was found or done, with the key facts (names, numbers, ids) that later answers may need.
Then a section "OPEN" with what is still pending or unanswered.
If an earlier summary is given, merge it: keep its requirements and facts unless they were replaced. Do not add anything that is not in the conversation."""


def head_tail(text: str, limit: int) -> str:
    """``text`` cut to about ``limit`` characters, keeping its start and its end."""
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return text[:head] + f"\n[... {len(text) - limit} characters cut to fit the context ...]\n" + text[-tail:]


def stub_for(name: str, status: str, content: str) -> str:
    """The short replacement for an old tool result."""
    preview = " ".join(content.split())[:STUB_PREVIEW_CHARS]
    return (
        f"[tool result: {name} | status: {status}]\n"
        f"[{count_tokens(content)} tokens, removed from context to save space; it began: {preview!r}. "
        "Call the tool again if you still need it.]"
    )


def _render_events(events: list[dict]) -> str:
    lines = []
    for e in events:
        kind = e["type"]
        if kind == "user":
            lines.append("USER: " + head_tail(e["content"], EVENT_CHARS))
        elif kind == "assistant":
            lines.append("ASSISTANT: " + head_tail(e["content"], EVENT_CHARS))
        elif kind == "tool_result":
            lines.append(f"TOOL RESULT ({e['name']}, {e['status']}): " + head_tail(e["content"], RESULT_CHARS))
        elif kind == "error":
            lines.append("NOTE: " + head_tail(e["content"], 300))
    return "\n".join(lines)


def _mechanical_summary(previous: str, events: list[dict]) -> str:
    """Used when the summarizing model fails: the user's own words, shortened,
    and the assistant's answers. No model needed, and the user's constraints
    are in it because they are in the user's messages."""
    lines = [previous] if previous else []
    for e in events:
        if e["type"] == "user":
            lines.append("USER REQUIREMENTS / REQUEST: " + head_tail(e["content"], 400))
    last_answers = [e["content"] for e in events if e["type"] == "assistant" and "<tool_call>" not in e["content"]]
    lines += ["ASSISTANT ANSWERED: " + head_tail(a, 200) for a in last_answers[-3:]]
    return "\n".join(lines)


def summarize(aux_lm: Any, previous: str, events: list[dict]) -> str:
    """One summary of ``events`` merged with the ``previous`` summary. Never
    longer than SUMMARY_MAX_TOKENS. BudgetExceeded propagates (the caller must
    see a spending stop); any other failure falls back to the mechanical summary."""
    body = ("EARLIER SUMMARY:\n" + previous + "\n\n" if previous else "") + "CONVERSATION TO COMPRESS:\n" + _render_events(events)
    try:
        text = aux_lm([{"role": "system", "content": SUMMARY_SYSTEM}, {"role": "user", "content": body}])
    except BudgetExceeded:
        raise
    except OutputTruncated as exc:
        text = exc.text or ""
    except Exception:  # a rate limit or a network error must not end the conversation
        text = ""
    text = (text or "").strip()
    if not text:
        text = _mechanical_summary(previous, events)
    if count_tokens(text) > SUMMARY_MAX_TOKENS:
        text = text[: SUMMARY_MAX_TOKENS * 4]  # the requirements come first, so they survive the cut
    return text
