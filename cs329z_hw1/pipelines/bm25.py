"""Problem 3 (bm25): BM25 ranking on an inverted index, with no retrieval library.

score(q, d) = sum over distinct query terms t of
    IDF(t) * f(t,d) * (k1 + 1) / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))
with k1 = 1.5, b = 0.75 and the Lucene IDF  ln((N - n_t + 0.5) / (n_t + 0.5) + 1).

Design:
* The index maps each term to its postings: parallel lists of document ids
  and term frequencies. It is built in one pass over the corpus.
* A document's length normalisation k1 * (1 - b + b * |d| / avgdl) does not
  depend on the query, so it is computed once per document at build time.
* A search touches only the postings of the query's own terms, so its cost
  grows with the number of documents that contain those terms and not with
  the size of the corpus. A document that shares no term with the query is
  never visited, and so is never returned.
* The index is never modified after it is built, so searches may run
  concurrently from several threads.
"""

from __future__ import annotations

import heapq
import math
from collections import Counter

from cs329z_hw1.tokenizer import tokenize

K1 = 1.5
B = 0.75


class BM25Index:
    __slots__ = ("num_docs", "postings", "norm", "idf")

    def __init__(self, docs: list[str]) -> None:
        postings: dict[str, tuple[list[int], list[int]]] = {}
        lengths: list[int] = []
        for doc_id, text in enumerate(docs):
            tokens = tokenize(text)
            lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                entry = postings.get(term)
                if entry is None:
                    postings[term] = ([doc_id], [tf])
                else:
                    entry[0].append(doc_id)
                    entry[1].append(tf)

        n = len(docs)
        avgdl = (sum(lengths) / n) if n else 0.0
        if avgdl == 0.0:  # every document is empty: nothing can match anyway
            avgdl = 1.0
        self.num_docs = n
        self.postings = postings
        self.norm = [K1 * (1 - B + B * length / avgdl) for length in lengths]
        # IDF of every term, from its document frequency (the length of its postings).
        self.idf = {
            term: math.log((n - len(ids) + 0.5) / (len(ids) + 0.5) + 1)
            for term, (ids, _) in postings.items()
        }

    def search(self, query: str, k: int) -> list[tuple[int, float]]:
        if k <= 0:
            return []
        scores: dict[int, float] = {}
        norm = self.norm
        for term in dict.fromkeys(tokenize(query)):  # distinct terms, in order
            entry = self.postings.get(term)
            if entry is None:
                continue
            ids, tfs = entry
            idf = self.idf[term]
            scale = idf * (K1 + 1)
            for doc_id, tf in zip(ids, tfs):
                scores[doc_id] = scores.get(doc_id, 0.0) + scale * tf / (tf + norm[doc_id])
        # Best score first; equal scores go to the lower document id.
        top = heapq.nsmallest(k, scores.items(), key=lambda item: (-item[1], item[0]))
        return [(doc_id, float(score)) for doc_id, score in top]


def build_index(docs: list[str]) -> BM25Index:
    return BM25Index(docs)


def search(index: BM25Index, query: str, k: int) -> list[tuple[int, float]]:
    return index.search(query, k)
