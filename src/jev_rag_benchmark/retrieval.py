from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from pathlib import Path

from .io import read_jsonl, write_jsonl
from .models import Candidate, Document

TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


class BM25Index:
    """Small deterministic BM25 index used to freeze candidate lists."""

    def __init__(self, documents: list[Document], *, k1: float = 1.5, b: float = 0.75):
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.doc_terms = [Counter(tokenize(f"{doc.title} {doc.text}")) for doc in documents]
        self.doc_lengths = [sum(terms.values()) for terms in self.doc_terms]
        self.avg_doc_length = sum(self.doc_lengths) / max(1, len(self.doc_lengths))
        self.df: Counter[str] = Counter()
        for terms in self.doc_terms:
            self.df.update(terms.keys())

    def retrieve(self, query: str, top_k: int = 20) -> list[Candidate]:
        query_terms = Counter(tokenize(query))
        n_docs = len(self.documents)
        scores: defaultdict[int, float] = defaultdict(float)
        for term, qtf in query_terms.items():
            df = self.df.get(term, 0)
            if not df:
                continue
            idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
            for idx, terms in enumerate(self.doc_terms):
                tf = terms.get(term, 0)
                if not tf:
                    continue
                norm = tf + self.k1 * (
                    1 - self.b + self.b * self.doc_lengths[idx] / max(self.avg_doc_length, 1e-9)
                )
                scores[idx] += qtf * idf * (tf * (self.k1 + 1) / norm)
        ranked = sorted(range(n_docs), key=lambda i: (-scores[i], self.documents[i].doc_id))
        return [
            Candidate(self.documents[idx], float(scores[idx]), rank)
            for rank, idx in enumerate(ranked[:top_k], start=1)
        ]

    def save_documents(self, path: str | Path) -> None:
        write_jsonl(path, self.documents)

    @classmethod
    def from_jsonl(cls, path: str | Path) -> "BM25Index":
        docs = [Document(**row) for row in read_jsonl(path)]
        return cls(docs)

