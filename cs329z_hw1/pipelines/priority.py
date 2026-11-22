"""Problem 1 (priority): label one email urgent / normal / ignore with one LLM call.

Design:
* One model call. The system prompt holds the full rubric and never changes,
  so the provider can cache it across emails; only the user message varies.
* The model is asked to end with two fixed lines, ``LABEL: <word>`` and
  ``REASON: <sentence>``, and the parser looks for those first.
* Replies that ignore the format fall back to a lenient read: a JSON-style
  ``"category"`` key, or a reply that names exactly one label word. Anything
  else (no label word, or several) is ``unknown``: we never guess.
* The email's weekday is computed and shown to the model, because the
  rubric's "two working days" depends on it and models get weekdays wrong.
"""

from __future__ import annotations

import re
from datetime import datetime

from cs329z_hw1.llm import BudgetExceeded
from cs329z_hw1.pipelines._lm import call_lm
from cs329z_hw1.types import Email, LMCallable

LABELS = ("urgent", "normal", "ignore")

# Long bodies (quoted threads, attachments pasted inline) add cost and say
# little about priority beyond their first few thousand characters.
MAX_BODY_CHARS = 6000
MAX_RECIPIENTS_SHOWN = 10

SYSTEM_PROMPT = """You triage email. For one email, decide what its recipient (the person in the To field) should do with it, judged at the time it was sent. Choose exactly one label.

urgent: BOTH must hold.
  1. The email asks the recipient to do something, or reports a live problem the recipient is responsible for (an outage, a failed payment, a customer or manager escalating an issue).
  2. The action is needed the same day, the next working day, or by a stated deadline at most two working days away. Working days are Monday to Friday, counted from the email's date.
  Examples: "The bank rejected this morning's wire. Approve the corrected one by 1:00 PM or we pay a late penalty." / "The customer's CFO is threatening to cancel. Can you call him before 4:00 today?"

normal: legitimate work or personal mail written for this recipient or a small working group, with no deadline in the next two working days. Status updates and FYIs, ordinary questions, documents for review without a near deadline, scheduling for later dates, replies in an ongoing discussion, personal notes from friends or family.
  Examples: "Attached is the draft term sheet. Comments before our meeting the week after next?" / "Thanks for lunch yesterday. Are you still coming to the lake in August?"

ignore: mail the recipient can leave unread at no cost. Advertising and vendor spam, newsletters and news digests, announcements to large employee lists (events, training, facilities notices), automated notifications that ask nothing of the recipient.
  Examples: "This week only: 40% off all standing desks." / "To all employees: the annual bake sale is in the lobby on Friday."

Deciding between labels:
- Wording does not decide the label. "URGENT" in a subject that asks nothing of the recipient is not urgent. A calm "I need your signature by 3:00 PM today" is urgent.
- A request with no deadline, or a deadline more than two working days away, is normal.
- Mass mailings and advertising are ignore even when they say "act now" or "important": nothing is asked of this recipient in particular.
- An automated notice asking this specific recipient to act (e.g. an expense report awaiting their approval) is normal, or urgent if it states a deadline within two working days.
- A short personal or social message from someone the recipient knows is normal, not ignore.
- Judge only the newest message; quoted older text is context.

Reply with exactly two lines and nothing else:
LABEL: <urgent|normal|ignore>
REASON: <one sentence explaining the label>"""


def _weekday(date: str) -> str:
    try:
        return datetime.fromisoformat(date.replace("Z", "+00:00")).strftime("%A")
    except (ValueError, AttributeError):
        return "unknown weekday"


def format_email(email: Email) -> str:
    to = list(email.get("to") or [])
    shown_to = ", ".join(to[:MAX_RECIPIENTS_SHOWN]) or "(none)"
    if len(to) > MAX_RECIPIENTS_SHOWN:
        shown_to += f", ... ({len(to)} recipients in total)"
    body = email.get("body") or ""
    if len(body) > MAX_BODY_CHARS:
        body = body[:MAX_BODY_CHARS] + "\n[... body truncated ...]"
    date = email.get("date") or ""
    return (
        f"From: {email.get('from') or '(unknown)'}\n"
        f"To: {shown_to}\n"
        f"Date: {date} ({_weekday(date)})\n"
        f"Subject: {email.get('subject') or '(no subject)'}\n\n"
        f"{body.strip() or '(empty body)'}"
    )


_LABEL_LINE = re.compile(r"(?im)^\W*(?:label|category|priority)\W*[:=]\W*(urgent|normal|ignore)\b")
_JSON_CATEGORY = re.compile(r'(?i)"(?:label|category)"\s*:\s*"(urgent|normal|ignore)"')
_REASON_LINE = re.compile(r"(?im)^\W*reason\W*[:=]\s*(.+)$")
_JSON_REASON = re.compile(r'(?i)"reason"\s*:\s*"([^"]+)"')
_ANY_LABEL = re.compile(r"(?i)\b(urgent|normal|ignore)\b")


def parse_reply(reply: str) -> dict:
    text = reply or ""
    category = "unknown"

    match = _LABEL_LINE.search(text) or _JSON_CATEGORY.search(text)
    if match:
        category = match.group(1).lower()
    else:
        found = {m.lower() for m in _ANY_LABEL.findall(text)}
        if len(found) == 1:  # several different label words: ambiguous, do not guess
            category = found.pop()

    reason_match = _REASON_LINE.search(text) or _JSON_REASON.search(text)
    reason = reason_match.group(1).strip() if reason_match else ""
    if not reason:
        reason = " ".join(text.split())[:300]
    if not reason:
        reason = "The model gave no reason."
    return {"category": category, "reason": reason}


def classify_priority(email: Email, lm: LMCallable) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": format_email(email)},
    ]
    try:
        reply = call_lm(lm, messages)  # retries transient provider errors
    except BudgetExceeded:
        raise  # the spending cap must stop the run, not become an 'unknown' label
    except Exception as exc:  # the contract says never raise
        return {"category": "unknown", "reason": f"Model call failed: {exc}"}
    return parse_reply(reply)
