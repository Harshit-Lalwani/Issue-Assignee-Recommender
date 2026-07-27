"""Ranking metrics for single-relevant-item retrieval (one true assignee per issue)."""
import math

import pandas as pd


def recall_at_k(ranked: list, true: str, k: int) -> float:
    return 1.0 if true in ranked[:k] else 0.0


def reciprocal_rank(ranked: list, true: str) -> float:
    try:
        rank = ranked.index(true) + 1
    except ValueError:
        return 0.0
    return 1.0 / rank


def ndcg_at_k(ranked: list, true: str, k: int) -> float:
    """Binary relevance, single relevant item -> IDCG=1, so this is 1/log2(rank+1)
    if the true item is within the top k, else 0."""
    try:
        rank = ranked[:k].index(true) + 1
    except ValueError:
        return 0.0
    return 1.0 / math.log2(rank + 1)


def evaluate(predictions: list[tuple[list, str]], ks=(5, 10)) -> dict:
    """predictions: list of (ranked_candidate_list, true_assignee_key).
    Returns mean Recall@k for each k, MRR, and NDCG@10."""
    n = len(predictions)
    if n == 0:
        return {}
    out = {}
    for k in ks:
        out[f"recall@{k}"] = sum(recall_at_k(r, t, k) for r, t in predictions) / n
    out["mrr"] = sum(reciprocal_rank(r, t) for r, t in predictions) / n
    out["ndcg@10"] = sum(ndcg_at_k(r, t, 10) for r, t in predictions) / n
    out["n"] = n
    return out


def leaderboard(rows: list[dict]) -> pd.DataFrame:
    """rows: list of {"project": ..., "method": ..., **metrics}"""
    df = pd.DataFrame(rows)
    cols = ["project", "method", "n", "recall@5", "recall@10", "mrr", "ndcg@10"]
    cols = [c for c in cols if c in df.columns]
    return df[cols].sort_values(["project", "method"])
