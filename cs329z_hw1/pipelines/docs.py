"""Problem 9 (search_docs): retrieval over the Cardinal Energy documents.

Two tools, two jobs:

* ``search_docs`` finds *where* to look: a few passages, each a short excerpt
  with its document and line range. It is a pointer, not the document.
* ``read_doc`` reads a bounded window of lines, so the model can follow a
  pointer, or read on, without a whole document entering its context.

Design choices (each is a memo answer):
* Passages, not whole documents, are what BM25 ranks. A 28,000-character
  handbook is one document but its answer is one paragraph; ranking passages
  puts that paragraph first and makes the snippet the evidence itself.
* A passage is a run of lines under one heading, at most ``CHUNK_CHARS``
  characters, split at blank lines when possible. The heading is stored with
  the passage (indexed and shown) because the heading is often what names the
  topic ("Meals") while the body only has the number.
* One result per document (its best passage). The corpus has near-duplicate
  documents (an expense policy for 2025 and for 2026, announcements that
  replace policy paragraphs); k results from one handbook would crowd out the
  later announcement that changed the answer.
* Every result carries the line range, so ``read_doc(doc_id, start_line)``
  can continue from it.
* The BM25 is the one from Problem 3 (``pipelines.bm25``).
"""

from __future__ import annotations

from dataclasses import dataclass

from cs329z_hw1.pipelines.bm25 import build_index, search
from cs329z_hw1.tokenizer import tokenize
from cs329z_hw1.tokens import count_tokens
from cs329z_hw1.types import DocWindow, SearchResult

CHUNK_CHARS = 420  # a snippet is this plus a short heading and line label: under 150 tokens
HEADING_CHARS = 80
WINDOW_TOKENS = 1400  # under the 1500-token limit with room for the count_tokens rounding


@dataclass
class Chunk:
    doc_id: str
    start_line: int
    end_line: int
    heading: str
    text: str


@dataclass
class DocIndex:
    docs: dict[str, dict]
    lines: dict[str, list[str]]
    chunks: list[Chunk]
    bm25: object


def _split_long(line: str) -> list[str]:
    """Pieces of at most CHUNK_CHARS characters, cut at spaces."""
    pieces = []
    while len(line) > CHUNK_CHARS:
        cut = line.rfind(" ", 0, CHUNK_CHARS)
        cut = cut if cut > CHUNK_CHARS // 2 else CHUNK_CHARS
        pieces.append(line[:cut].strip())
        line = line[cut:].strip()
    if line:
        pieces.append(line)
    return pieces


def chunk_document(doc_id: str, text: str) -> list[Chunk]:
    chunks: list[Chunk] = []
    heading = ""
    pending: list[tuple[int, str]] = []  # (line number, text)

    def flush() -> None:
        if pending:
            body = "\n".join(t for _, t in pending).strip()
            if tokenize(body):
                chunks.append(Chunk(doc_id, pending[0][0], pending[-1][0], heading, body))
        pending.clear()

    for number, line in enumerate(text.splitlines(), start=1):
        if line.startswith("#"):
            flush()
            heading = line.lstrip("#").strip()[:HEADING_CHARS]
            continue
        if not line.strip():
            if sum(len(t) for _, t in pending) > CHUNK_CHARS // 2:
                flush()  # a blank line is a good place to cut once the passage has some size
            continue
        for piece in _split_long(line):
            if pending and sum(len(t) + 1 for _, t in pending) + len(piece) > CHUNK_CHARS:
                flush()
            pending.append((number, piece))
    flush()
    return chunks


def build_doc_index(docs: list[dict]) -> DocIndex:
    by_id = {d["doc_id"]: d for d in docs}
    lines = {d["doc_id"]: d["text"].splitlines() for d in docs}
    chunks = [c for d in docs for c in chunk_document(d["doc_id"], d["text"])]
    bm25 = build_index([f"{c.heading}\n{c.text}" for c in chunks])
    return DocIndex(by_id, lines, chunks, bm25)


def _snippet(chunk: Chunk) -> str:
    label = f"(lines {chunk.start_line}-{chunk.end_line}) "
    head = f"{chunk.heading}: " if chunk.heading else ""
    return label + head + " ".join(chunk.text.split())


def search_documents(index: DocIndex, query: str, k: int = 5) -> list[SearchResult]:
    if k < 1 or not tokenize(query):
        return []
    results: list[dict] = []
    seen: set[str] = set()
    # the scores come best first; take each document's best passage once
    for chunk_id, _score in search(index.bm25, query, len(index.chunks)):
        chunk = index.chunks[chunk_id]
        if chunk.doc_id in seen:
            continue
        seen.add(chunk.doc_id)
        results.append(
            {
                "doc_id": chunk.doc_id,
                "title": index.docs[chunk.doc_id]["title"],
                "snippet": _snippet(chunk),
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
            }
        )
        if len(results) == k:
            break
    return results


def read_window(index: DocIndex, doc_id: str, start_line: int = 1, max_tokens: int = WINDOW_TOKENS) -> DocWindow:
    if doc_id not in index.lines:
        raise KeyError(f"no document with doc_id {doc_id!r}; doc_ids come from search_docs")
    lines = index.lines[doc_id]
    total = len(lines)
    if isinstance(start_line, bool) or not isinstance(start_line, int) or not 1 <= start_line <= total:
        raise ValueError(f"start_line must be between 1 and {total} for {doc_id}, got {start_line!r}")
    end = start_line
    used = count_tokens(lines[start_line - 1])
    while end < total:
        used += count_tokens("\n" + lines[end])
        if used > max_tokens:
            break
        end += 1
    return {
        "doc_id": doc_id,
        "start_line": start_line,
        "end_line": end,
        "total_lines": total,
        "text": "\n".join(lines[start_line - 1 : end]),
    }
