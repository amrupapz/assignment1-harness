"""Problem 2 (daily_digest): a morning summary of one day of email.

Pipeline (the control flow is code; the model classifies and writes):
1. Classify every email with the priority pipeline, concurrently.
2. Treat ``unknown`` as ``normal``; drop ``ignore``. Ignored emails are never
   shown to the writing call, so nothing from them can leak into the digest.
3. One item per thread: the thread's most urgent email, latest on ties.
4. Order urgent first, then normal (latest first within a label).
5. One writing call. Urgent items carry an excerpt of the body so the model
   can name the company, project or site; normal items carry only a line.
   The word limit comes from the prompt. If the model overshoots, one more
   call asks it to rewrite shorter (the text is never cut by code).
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from cs329z_hw1.llm import BudgetExceeded
from cs329z_hw1.pipelines._lm import call_lm
from cs329z_hw1.pipelines.priority import classify_priority
from cs329z_hw1.types import Email, LMCallable

MAX_WORDS = 300
TARGET_WORDS = 220  # asked for in the prompt, leaving a margin under MAX_WORDS
MAX_WORKERS = 16
MAX_NORMAL_ITEMS = 40  # a full day has ~280 emails; keep the writing prompt small
URGENT_EXCERPT_CHARS = 700
RANK = {"urgent": 0, "normal": 1}

SYSTEM_PROMPT = f"""You write a morning email digest for a busy employee. You receive the day's emails that need attention, already triaged into URGENT and NORMAL items, one item per thread.

Rules:
- At most {TARGET_WORDS} words in total. This is a hard limit; be terse.
- Start with the urgent items. Cover EVERY urgent item, one bullet each, and name the specific company, project, site, deal, or asset it concerns, exactly as spelled in the email (for example "Orlin Valley wire", not "a wire transfer"). Say what is needed and by when.
- Then summarize the normal items briefly, grouped by theme. Skip minor ones if space is short; never skip an urgent one.
- Mention only what appears in the items you are given. Do not invent details.
- Plain text with short bullets. No preamble, no sign-off."""


def _clean(text: str, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit] + " ..."


# One cap shared by every digest running at once (the live test runs five
# days in parallel; the agent may run several tools). Without it, 5 x 16
# simultaneous calls got rate-limited and failed classifications came back
# "unknown", which silently demoted urgent emails to normal.
_CALL_SLOTS = threading.BoundedSemaphore(MAX_WORKERS)


def _classify_one(email: Email, lm: LMCallable) -> dict:
    with _CALL_SLOTS:
        return classify_priority(email, lm)


def _classify_all(emails: list[Email], lm: LMCallable) -> list[dict]:
    if not emails:
        return []
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(emails))) as pool:
        return list(pool.map(lambda e: _classify_one(e, lm), emails))


def select_items(emails: list[Email], labels: list[dict]) -> list[tuple[Email, str, str]]:
    """(email, label, reason) per thread, ignore dropped, urgent first."""
    best: dict[str, tuple[Email, str, str]] = {}
    for email, result in zip(emails, labels):
        label = result["category"]
        if label == "ignore":
            continue
        if label not in RANK:  # unknown is treated like normal
            label = "normal"
        key = email.get("thread_id") or email.get("id") or str(id(email))
        item = (email, label, result["reason"])
        current = best.get(key)
        if current is None or (RANK[label], _neg_date(email)) < (RANK[current[1]], _neg_date(current[0])):
            best[key] = item
    items = list(best.values())
    items.sort(key=lambda it: (RANK[it[1]], _neg_date(it[0])))
    return items


def _neg_date(email: Email) -> tuple:
    # Sort key that puts later emails first (ISO dates compare as strings).
    return tuple(-ord(c) for c in (email.get("date") or ""))


def _render_items(items: list[tuple[Email, str, str]]) -> str:
    urgent = [it for it in items if it[1] == "urgent"]
    normal = [it for it in items if it[1] == "normal"][:MAX_NORMAL_ITEMS]
    lines = [f"URGENT ITEMS ({len(urgent)}):"]
    for n, (email, _, reason) in enumerate(urgent, 1):
        lines.append(
            f"[U{n}] From: {email.get('from') or '?'} | Subject: {email.get('subject') or '(no subject)'}\n"
            f"     Why urgent: {reason}\n"
            f"     Excerpt: {_clean(email.get('body') or '', URGENT_EXCERPT_CHARS)}"
        )
    if not urgent:
        lines.append("(none)")
    lines.append(f"\nNORMAL ITEMS ({len(normal)}):")
    for n, (email, _, reason) in enumerate(normal, 1):
        lines.append(
            f"[N{n}] {email.get('subject') or '(no subject)'} (from {email.get('from') or '?'}): "
            f"{_clean(reason, 200)}"
        )
    if not normal:
        lines.append("(none)")
    return "\n".join(lines)


def _fallback(items: list[tuple[Email, str, str]]) -> str:
    """Used only if the writing call fails or returns nothing."""
    urgent = [it for it in items if it[1] == "urgent"]
    normal_count = sum(1 for it in items if it[1] == "normal")
    parts = [f"Urgent: {(e.get('subject') or '(no subject)')} - {_clean(r, 120)}" for e, _, r in urgent]
    parts.append(f"{normal_count} other thread(s) need no action in the next two working days.")
    return "\n".join(parts)


def daily_digest(emails: list[Email], lm: LMCallable, writer_lm: LMCallable | None = None) -> str:
    """``lm`` classifies; ``writer_lm`` (default ``lm``) writes the digest."""
    writer_lm = writer_lm or lm
    if not emails:
        return "No emails arrived on this day."

    items = select_items(emails, _classify_all(emails, lm))
    if not items:
        return f"Nothing needs your attention: all {len(emails)} emails were newsletters, ads or mass announcements."

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _render_items(items)},
    ]
    try:
        digest = call_lm(writer_lm, messages).strip()
        if len(digest.split()) > MAX_WORDS:
            messages += [
                {"role": "assistant", "content": digest},
                {
                    "role": "user",
                    "content": f"That is {len(digest.split())} words. Rewrite it in at most {TARGET_WORDS} words. "
                    "Keep every urgent item and its company, project or site name; shorten or drop normal items.",
                },
            ]
            digest = call_lm(writer_lm, messages).strip()
    except BudgetExceeded:
        raise  # do not hide a spending-cap stop behind the canned fallback
    except Exception:
        return _fallback(items)
    return digest or _fallback(items)
