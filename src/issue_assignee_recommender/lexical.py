"""Sparse BM25 index.

Same scoring function as the `rank_bm25.BM25Okapi` used by the Phase 1 baseline, but with the
document-side weights precomputed into a sparse term-document matrix so a query costs one
sparse matvec instead of a full pass over the corpus in Python. That matters here because
Phase 3 needs BM25 for *every* candidate-generation call (tens of thousands of queries), not
just the 856-4,680 of a single baseline evaluation.

Kept separate from `baselines.py` on purpose: the Phase 1 leaderboard numbers were produced
with `BM25Okapi` and are frozen, so that code path is left untouched.
"""
from collections import Counter

import numpy as np
import scipy.sparse as sp

from .text import tokenize


class SparseBM25:
    def __init__(self, corpus_tokens: list[list[str]], k1: float = 1.5, b: float = 0.75):
        vocab: dict[str, int] = {}
        rows, cols, tf = [], [], []
        doc_len = np.zeros(len(corpus_tokens), dtype="float32")
        for i, toks in enumerate(corpus_tokens):
            doc_len[i] = len(toks)
            for term, count in Counter(toks).items():
                j = vocab.setdefault(term, len(vocab))
                rows.append(i)
                cols.append(j)
                tf.append(count)
        n_docs = len(corpus_tokens)
        self.n_docs = n_docs
        self.vocab = vocab
        if n_docs == 0 or not vocab:
            self.matrix = sp.csc_matrix((max(n_docs, 0), 0), dtype="float32")
            return

        tf_arr = np.asarray(tf, dtype="float32")
        avgdl = float(doc_len.mean()) or 1.0
        norm = k1 * (1 - b + b * doc_len[np.asarray(rows)] / avgdl)
        weighted = tf_arr * (k1 + 1) / (tf_arr + norm)
        matrix = sp.csr_matrix((weighted, (rows, cols)), shape=(n_docs, len(vocab)))
        df = np.asarray((matrix > 0).sum(axis=0)).ravel()
        idf = np.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)
        self.matrix = matrix.multiply(sp.csr_matrix(idf.astype("float32"))).tocsc()

    def search(self, text: str, k: int) -> list[tuple[int, float]]:
        """Top-k (doc_index, score) for a raw query string, strictly positive scores only."""
        if self.n_docs == 0 or not self.vocab:
            return []
        counts = Counter(t for t in tokenize(text) if t in self.vocab)
        if not counts:
            return []
        cols = [self.vocab[t] for t in counts]
        weights = np.asarray(list(counts.values()), dtype="float32")
        scores = np.asarray(self.matrix[:, cols].dot(weights)).ravel()
        k = min(k, self.n_docs)
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[scores[top] > 0]
        return sorted(((int(i), float(scores[i])) for i in top), key=lambda x: -x[1])
