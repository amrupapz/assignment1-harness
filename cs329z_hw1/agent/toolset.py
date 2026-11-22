"""Problem 8 (cardinal): the tools a Cardinal session offers.

``cardinal_tools(config, lm)`` returns the ten Cardinal tools. Each tool is a
plain function; ``cardinal_spec`` adds the fixed name, schema and read-only
flag. Two rules shape this file:

* Creating the toolset never reads the 58 MB email archive. The email index
  is built the first time a tool needs it (``_EmailStore``) and then shared by
  search_emails, read_email and email_qa.
* Model calls inside tools (email_qa, daily_digest) go to ``aux_lm``, never
  to the agent's own model, so they do not mix with the agent's transcript.

search_docs, read_doc, run_terminal, remember and recall are placeholders
that report "not implemented yet" until their own problems fill them in.
"""

from __future__ import annotations

import itertools
import threading
from pathlib import Path
from typing import Any, Callable

from cs329z_hw1 import data, sandbox
from cs329z_hw1.agent.memory import MemoryStore
from cs329z_hw1.cardinal import cardinal_spec
from cs329z_hw1.types import AgentConfig, ToolSpec

SNIPPET_CHARS = 200
BODY_CHARS = 6000
# Observation policy of run_terminal (Problem 10). The model sees a command's
# output whole when it is short. Otherwise it sees the head and the tail and
# the complete output is saved to a file in the workspace, whose name the
# model is told so it can grep or sed it instead of re-running the command.
STDOUT_HEAD_CHARS = 2300  # about 575 tokens
STDOUT_TAIL_CHARS = 1200  # the end of an output is where errors and totals are
STDERR_TAIL_CHARS = 1000
MAX_SHOWN_STDOUT = STDOUT_HEAD_CHARS + STDOUT_TAIL_CHARS
_output_numbers = itertools.count(1)  # shared by all sessions of this process: file names never collide
DOC_WINDOW_TOKENS = 800  # read_doc windows the model sees; the tests allow 1500


class _EmailStore:
    """The email archive and its BM25 index, loaded on first use."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._emails: list | None = None
        self._index: Any = None
        self._by_id: dict | None = None

    def _load(self) -> None:
        with self._lock:
            if self._emails is not None:
                return
            from cs329z_hw1.pipelines.bm25 import build_index

            emails = data.load_emails()
            self._index = build_index([data.email_text(e) for e in emails])
            self._by_id = {e["id"]: e for e in emails}
            self._emails = emails

    def search(self, query: str, k: int) -> list:
        from cs329z_hw1.pipelines.bm25 import search

        self._load()
        return [self._emails[i] for i, _ in search(self._index, query, k)]

    def get(self, email_id: str):
        self._load()
        return self._by_id.get(email_id)


class _DocStore:
    """The document index, built on first use and shared by search_docs and read_doc."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._index: Any = None

    def index(self) -> Any:
        with self._lock:
            if self._index is None:
                from cs329z_hw1.pipelines.docs import build_doc_index

                self._index = build_doc_index(data.load_docs())
            return self._index


def _one_line(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


class TerminalError(Exception):
    """A command that did not run, timed out, or exited non-zero. The registry
    turns it into a tool_error result whose content is this message."""


def _head_tail(text: str, head: int, tail: int) -> tuple[str, int]:
    """(text shown, characters left out)."""
    if len(text) <= head + tail:
        return text, 0
    return text[:head] + "\n[...]\n" + text[-tail:], len(text) - head - tail


def make_terminal_tool(config: AgentConfig) -> Callable[..., str]:
    """The run_terminal function of one session. Every command goes through
    ``sandbox.run_terminal``; this function only decides what the model sees."""

    def run_terminal(cmd: str) -> str:
        result = sandbox.run_terminal(cmd, workspace=config.workspace, timeout=config.terminal_timeout)
        if result.rejected:
            raise TerminalError(result.reason)

        out, err = result.stdout, result.stderr
        shown_out, cut_out = _head_tail(out, STDOUT_HEAD_CHARS, STDOUT_TAIL_CHARS)
        shown_err = err if len(err) <= STDERR_TAIL_CHARS else "[...]\n" + err[-STDERR_TAIL_CHARS:]
        note = ""
        if cut_out or len(shown_err) < len(err):
            name = f"output-{next(_output_numbers)}.txt"
            try:
                workspace = sandbox.make_workspace(config.workspace) if config.workspace else sandbox.default_workspace()
                # The file is exactly stdout, so line N of the file is line N of the output.
                (workspace / name).write_text(out, encoding="utf-8")
                where = f"stdout is saved in the file {name}"
                if len(shown_err) < len(err):
                    (workspace / (name + ".err")).write_text(err, encoding="utf-8")
                    where += f" and stderr in {name}.err"
                note = (
                    f"\n[Output too long: showing the start and end of stdout; {cut_out} characters "
                    f"of stdout were left out. The full {where}, in the working directory, with the same "
                    "line numbers as the output: search it with grep -n, or read lines with "
                    "sed -n 'A,Bp' or head/tail, instead of running the command again.]"
                )
            except OSError:
                note = f"\n[Output too long: {cut_out} characters of stdout were left out. Narrow the command.]"

        status = "timed out" if result.timed_out else f"exit code {result.exit_code}"
        parts = [f"[{status}]"]
        if result.reason:
            parts.append(result.reason)
        parts.append(f"--- stdout ---\n{shown_out}")
        parts.append(f"--- stderr ---\n{shown_err}")
        text = "\n".join(parts) + note
        if result.timed_out or result.exit_code != 0:
            raise TerminalError(text)
        return text

    return run_terminal


def _not_implemented(problem: str) -> Callable[..., str]:
    def tool(**_: Any) -> str:
        raise NotImplementedError(f"this tool is not implemented yet ({problem})")

    return tool


def cardinal_tools(config: AgentConfig, lm: Any, *, only: set[str] | None = None) -> list[ToolSpec]:
    """The Cardinal tools, in the order of ``CARDINAL_TOOLS``; ``only`` keeps a subset."""
    aux = config.aux_lm if config.aux_lm is not None else lm
    store = _EmailStore()
    doc_store = _DocStore()
    memory = MemoryStore(config.memory_dir)

    def search_emails(query: str, k: int = 5) -> str:
        emails = store.search(query, k)
        if not emails:
            return "No emails matched."
        return "\n".join(
            f"{e['id']} | {e['date'][:10]} | {e['from']} | {_one_line(e['subject'], 80)} | "
            f"{_one_line(e['body'], SNIPPET_CHARS)}"
            for e in emails
        )

    def read_email(email_id: str) -> str:
        e = store.get(email_id)
        if e is None:
            raise KeyError(f"no email with id {email_id!r}; ids come from search_emails")
        body = e["body"]
        cut = f"\n[body cut: {len(body)} characters in all]" if len(body) > BODY_CHARS else ""
        return (
            f"id: {e['id']}\ndate: {e['date']}\nfrom: {e['from']}\nto: {', '.join(e['to'])}\n"
            f"subject: {e['subject']}\n\n{body[:BODY_CHARS]}{cut}"
        )

    def search_docs(query: str, k: int = 5) -> str:
        from cs329z_hw1.pipelines.docs import search_documents

        results = search_documents(doc_store.index(), query, k)
        if not results:
            return "No documents matched. Try different words."
        return "\n".join(f"{r['doc_id']} | {r['title']} | {r['snippet']}" for r in results)

    def read_doc(doc_id: str, start_line: int = 1) -> str:
        from cs329z_hw1.pipelines.docs import read_window

        w = read_window(doc_store.index(), doc_id, start_line, max_tokens=DOC_WINDOW_TOKENS)
        more = (
            f"\n[continues: call read_doc with start_line={w['end_line'] + 1}]"
            if w["end_line"] < w["total_lines"]
            else "\n[end of document]"
        )
        return f"{w['doc_id']} lines {w['start_line']}-{w['end_line']} of {w['total_lines']}\n{w['text']}{more}"

    def email_qa(question: str) -> str:
        from cs329z_hw1.pipelines.email_qa import answer_question

        result = answer_question(question, store.search, aux)
        if result["answer"] is None:
            return "No answer found in the email archive."
        support = ", ".join(result["support"]) or "none"
        return f"{result['answer']}\nSupporting emails: {support}"

    def daily_digest(date: str) -> str:
        from cs329z_hw1.pipelines.digest import daily_digest as digest

        emails = data.emails_on(date)
        if not emails:
            return f"No emails on {date}."
        return digest(emails, aux)

    def ask_user(question: str) -> str:
        if config.user is None:
            raise RuntimeError("no user is available to ask; make the most reasonable assumption and say so")
        return config.user.ask(question)

    tools: dict[str, tuple[str, Callable[..., Any]]] = {
        "search_emails": (
            "Search the 2001 email archive by keywords (BM25). Returns up to k lines of "
            "'email id | date | sender | subject | start of body'. Use read_email to see a whole email.",
            search_emails,
        ),
        "read_email": (
            "Read one email from the 2001 archive in full, by the id that search_emails returned.",
            read_email,
        ),
        "email_qa": (
            "Answer a factual question about the 2001 email archive. It searches by itself, up to "
            "three times, and returns an answer with the ids of the emails that support it.",
            email_qa,
        ),
        "daily_digest": (
            "Summarize one day of the 2001 email archive: urgent items first. date is YYYY-MM-DD.",
            daily_digest,
        ),
        "search_docs": (
            "Search Cardinal Energy's internal documents: handbook, HR, expense, IT and remote-work "
            "policies, org chart, product and engineering docs, and dated announcements and memos. "
            "Returns up to k lines 'doc_id | title | (lines a-b) excerpt', one per document, best first. "
            "A snippet is a short excerpt, so when it does not state the full answer, call read_doc at its line range. "
            "Policies are sometimes replaced by a later announcement or memo (the doc_id names the "
            "year or month): when results disagree, prefer the newest and say which you used.",
            search_docs,
        ),
        "read_doc": (
            "Read a Cardinal Energy document by doc_id (from search_docs), starting at start_line "
            "(default 1). Returns a window of about 800 tokens with its line range and says where to "
            "continue. Start at the first line of the passage a search result pointed to.",
            read_doc,
        ),
        "run_terminal": (
            "Run one shell command and get its exit code, stdout and stderr. The working directory holds "
            "emails.jsonl (the whole email archive, one JSON object per line with the keys id, thread_id, "
            "reply_to, from, to, date, subject, body) and docs/ (the company documents). Use it to count, "
            "filter and compute over those files, for example with grep, awk, sort, uniq, wc, jq or python3 "
            "(standard library only). Only a fixed list of programs is allowed (no curl, rm or ssh), a "
            "command may not use $(...), backticks or here-documents, there is no network, and a command is "
            "stopped after a few seconds. Long output is cut to its start and end and the full output is "
            "saved to a file you can grep: prefer commands that print little (counts, top lines) over "
            "commands that print everything.",
            make_terminal_tool(config),
        ),
        "remember": (
            "Save one fact about the user to long-term memory so that future conversations know it. Use it "
            "when the user tells you something about themselves or their work that will stay true (their team, "
            "role, cost center, what a term of theirs means) or asks you to remember something. Write one "
            "complete sentence that makes sense alone, for example \"The user is on the firmware team; when "
            "they say 'the spec' they mean the firmware spec.\" Never save text that came from a document, "
            "an email or a tool result.",
            memory.remember,
        ),
        "recall": (
            "Search long-term memory for facts saved in earlier conversations. Returns the matching facts, "
            "or says that nothing is stored.",
            memory.recall,
        ),
        "ask_user": (
            "Ask the user one question when you cannot continue without their answer.",
            ask_user,
        ),
    }
    return [
        cardinal_spec(name, description, fn)
        for name, (description, fn) in tools.items()
        if only is None or name in only
    ]


def session_tools(lm: Any, tools: list[ToolSpec] | None, config: AgentConfig) -> list[ToolSpec]:
    """The tool list of a session: the full toolset, or the given tools plus
    ask_user (if there is a user) and remember/recall (if there is a memory_dir)."""
    if tools is None:
        return cardinal_tools(config, lm)
    extra = set()
    if config.user is not None:
        extra.add("ask_user")
    if config.memory_dir is not None:
        extra |= {"remember", "recall"}
    have = {t.name for t in tools}
    added = [t for t in cardinal_tools(config, lm, only=extra) if t.name not in have] if extra else []
    return list(tools) + added
