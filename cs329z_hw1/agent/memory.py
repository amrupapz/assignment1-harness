"""Problem 12 (memory): facts that outlive a session.

The store is one JSON file, ``<memory_dir>/facts.json``: a list of
``{"id": n, "fact": text}``. Two ways in:

* The prompt. A new session starts with the stored facts in its system prompt
  (``prompt_section``), newest first, up to ``PROMPT_MAX_TOKENS``. So a fact
  the user stated last week is simply known, without the model having to
  think of calling ``recall`` first. When the store is larger than that, the
  prompt says so and ``recall`` is how the rest is reached.
* The ``recall`` tool: BM25 over the facts (the Problem 3 index). While the
  store is small (``RETURN_ALL_BELOW`` facts), it returns all of them, best
  match first, because a vague query ("what do you know about me") should not
  come back empty.

A fact is stored as one self-contained sentence, so that it still means
something when read alone in a later session. Exact duplicates are not
stored twice. The file is written to a temporary name and renamed, so a crash
cannot leave half a file, and a lock keeps concurrent sessions from losing
each other's facts.
"""

from __future__ import annotations

import json
import os
import re
import threading
from pathlib import Path

from cs329z_hw1.pipelines.bm25 import build_index, search
from cs329z_hw1.tokens import count_tokens

FILE_NAME = "facts.json"
PROMPT_MAX_TOKENS = 700
MAX_FACT_CHARS = 600
RETURN_ALL_BELOW = 9
RECALL_TOP_K = 5

_STOP = {"the", "is", "are", "a", "an", "of", "to", "and", "in", "on", "s", "it", "that", "for", "with", "as"}
SAME_FACT_OVERLAP = 0.5  # share of words two facts must have in common to be about the same thing


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP}


def _same_subject(a: str, b: str) -> bool:
    wa, wb = _words(a), _words(b)
    return bool(wa and wb) and len(wa & wb) / len(wa | wb) > SAME_FACT_OVERLAP


_locks: dict[str, threading.Lock] = {}
_locks_guard = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    with _locks_guard:
        return _locks.setdefault(str(path.resolve()), threading.Lock())


class MemoryStore:
    def __init__(self, directory: str | Path | None) -> None:
        self.dir = Path(directory) if directory is not None else None

    @property
    def enabled(self) -> bool:
        return self.dir is not None

    @property
    def path(self) -> Path:
        return self.dir / FILE_NAME

    def load(self) -> list[dict]:
        if not self.enabled or not self.path.exists():
            return []
        try:
            facts = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return []  # a damaged file is treated as an empty store, not as a crash
        return [f for f in facts if isinstance(f, dict) and isinstance(f.get("fact"), str)]

    def remember(self, fact: str) -> str:
        if not self.enabled:
            raise RuntimeError("memory is not available in this session")
        fact = " ".join(str(fact).split())
        if not fact:
            raise ValueError("fact is empty; write one complete sentence")
        if len(fact) > MAX_FACT_CHARS:
            raise ValueError(f"fact is too long ({len(fact)} characters, limit {MAX_FACT_CHARS}); store one short fact")
        self.dir.mkdir(parents=True, exist_ok=True)
        with _lock_for(self.path):
            facts = self.load()
            if any(" ".join(f["fact"].split()).lower() == fact.lower() for f in facts):
                return "Already stored."
            # A correction ("my number is 31908, not 30918") says nearly the same thing as the fact it
            # corrects, so it replaces that fact instead of sitting beside it: a later session must
            # find one number, not two to choose between.
            old = [f for f in facts if _same_subject(f["fact"], fact)]
            facts = [f for f in facts if f not in old]
            facts.append({"id": max((f.get("id", 0) for f in facts + old), default=0) + 1, "fact": fact})
            tmp = self.path.with_suffix(".tmp")
            tmp.write_text(json.dumps(facts, indent=2, ensure_ascii=False), encoding="utf-8")
            os.replace(tmp, self.path)
        replaced = f" It replaced {len(old)} earlier fact(s) about the same thing." if old else ""
        return f"Stored.{replaced} Memory now holds {len(facts)} fact(s)."

    def recall(self, query: str) -> str:
        if not self.enabled:
            raise RuntimeError("memory is not available in this session")
        facts = self.load()
        if not facts:
            return "Memory is empty: nothing has been remembered yet."
        ranked = [facts[i] for i, _ in search(build_index([f["fact"] for f in facts]), str(query), len(facts))]
        if len(facts) < RETURN_ALL_BELOW:
            ranked += [f for f in facts if f not in ranked]  # best matches first, then the rest
        else:
            ranked = ranked[:RECALL_TOP_K]
        if not ranked:
            return "No stored fact matched that query."
        return "\n".join(f"- {f['fact']}" for f in ranked)

    def prompt_section(self) -> str:
        """What a new session starts with: the facts, newest first, within the
        token cap. Empty if there is nothing stored (or no memory_dir)."""
        facts = self.load()
        if not facts:
            return ""
        lines, used = [], 0
        for f in reversed(facts):
            line = f"- {f['fact']}"
            used += count_tokens(line) + 1
            if used > PROMPT_MAX_TOKENS:
                break
            lines.append(line)
        more = len(facts) - len(lines)
        note = f"\n({more} older fact(s) are stored too: use recall to search them.)" if more else ""
        return (
            "What you remember from earlier conversations with this user (use it without asking again; if two facts disagree, the one listed first is newer and right, so never give the user both):\n"
            + "\n".join(lines)
            + note
        )
