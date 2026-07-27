"""Phase 2: semantic retrieval with sentence-transformer embeddings + a flat (exact) FAISS
index. At this dataset's scale (tens of thousands of vectors per project, well under the
200k the HANDOFF sizes for) an exact flat index is fast enough that IVF/HNSW approximate
indexes would only add tuning surface for zero benefit -- skipped on purpose.
"""
from collections import Counter

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .baselines import PopularityBaseline

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class EmbeddingRetriever:
    name = "embedding_faiss"

    def __init__(self, model: SentenceTransformer, k_neighbors: int = 50):
        self.model = model
        self.k_neighbors = k_neighbors

    def fit(self, train_labeled):
        self.rows = train_labeled.reset_index(drop=True)
        texts = [
            f"{r.summary or ''} {r.description or ''}" for r in self.rows.itertuples()
        ]
        emb = self.model.encode(
            texts, batch_size=256, show_progress_bar=False, convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb)
        self.assignees = self.rows["assignee_key"].tolist()
        self.fallback = PopularityBaseline()
        self.fallback.fit(train_labeled)

    def encode_query(self, text: str) -> np.ndarray:
        return self.model.encode(
            [text], convert_to_numpy=True, normalize_embeddings=True
        ).astype("float32")

    def rank_from_vector(self, qvec: np.ndarray, top_n=50):
        scores, idx = self.index.search(qvec, self.k_neighbors)
        agg = Counter()
        for score, i in zip(scores[0], idx[0]):
            if i < 0 or score <= 0:
                continue
            agg[self.assignees[i]] += float(score)
        if not agg:
            return self.fallback.ranking[:top_n]
        ranked = [d for d, _ in agg.most_common()]
        for d in self.fallback.ranking:
            if d not in ranked:
                ranked.append(d)
            if len(ranked) >= top_n:
                break
        return ranked[:top_n]

    def rank(self, issue_row, top_n=50):
        text = f"{issue_row.get('summary') or ''} {issue_row.get('description') or ''}"
        if not text.strip():
            return self.fallback.ranking[:top_n]
        qvec = self.encode_query(text)
        return self.rank_from_vector(qvec, top_n)

    def rank_batch(self, rows, top_n=50):
        """Batch-encode queries for throughput; rows is a DataFrame of test issues."""
        texts = [f"{r.summary or ''} {r.description or ''}" for r in rows.itertuples()]
        embs = self.model.encode(
            texts, batch_size=256, show_progress_bar=False, convert_to_numpy=True,
            normalize_embeddings=True,
        ).astype("float32")
        scores, idxs = self.index.search(embs, self.k_neighbors)
        out = []
        for row_scores, row_idx in zip(scores, idxs):
            agg = Counter()
            for score, i in zip(row_scores, row_idx):
                if i < 0 or score <= 0:
                    continue
                agg[self.assignees[i]] += float(score)
            if not agg:
                out.append(self.fallback.ranking[:top_n])
                continue
            ranked = [d for d, _ in agg.most_common()]
            for d in self.fallback.ranking:
                if d not in ranked:
                    ranked.append(d)
                if len(ranked) >= top_n:
                    break
            out.append(ranked[:top_n])
        return out
