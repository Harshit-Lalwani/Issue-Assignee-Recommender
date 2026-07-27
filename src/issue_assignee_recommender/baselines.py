"""Phase 1 baselines, per HANDOFF:

1. Most-active-developer (pure popularity)
2. Component-owner heuristic
3. BM25 lexical similarity to past issues, kNN over their resolvers

All fit() calls take only train-period, labeled (resolved+assigned) issues -- consistent with
the leakage rule that nothing after issue creation may inform a *test* issue's ranking, and
"who resolved it" is exactly the kind of post-creation information a real deployment would only
have for train-period issues.
"""
from collections import Counter, defaultdict

from rank_bm25 import BM25Okapi

from .text import tokenize


def _as_list(x):
    """Components/labels come back as numpy arrays from parquet list columns; None/NaN
    and empty arrays should both just mean 'no components'."""
    if x is None:
        return []
    try:
        if len(x) == 0:
            return []
    except TypeError:
        return []
    return list(x)


class PopularityBaseline:
    """Rank every issue identically: by how many issues each developer has resolved in train."""

    name = "popularity"

    def fit(self, train_labeled):
        self.ranking = [d for d, _ in Counter(train_labeled["assignee_key"]).most_common()]

    def rank(self, issue_row, top_n=50):
        return self.ranking[:top_n]


class ComponentOwnerBaseline:
    """Rank by how often a developer resolved issues sharing this issue's component(s).
    Falls back to global popularity for issues with no components, or components unseen
    in train (both real cold-start situations, not bugs)."""

    name = "component_owner"

    def fit(self, train_labeled):
        self.component_counts: dict[str, Counter] = defaultdict(Counter)
        for _, row in train_labeled.iterrows():
            for comp in _as_list(row["components"]):
                self.component_counts[comp][row["assignee_key"]] += 1
        self.fallback = PopularityBaseline()
        self.fallback.fit(train_labeled)

    def rank(self, issue_row, top_n=50):
        components = _as_list(issue_row["components"])
        agg = Counter()
        for comp in components:
            agg.update(self.component_counts.get(comp, {}))
        if not agg:
            return self.fallback.rank(issue_row, top_n)
        ranked = [d for d, _ in agg.most_common()]
        # pad with popularity fallback so ties / short lists still fill top_n candidates
        for d in self.fallback.ranking:
            if d not in ranked:
                ranked.append(d)
            if len(ranked) >= top_n:
                break
        return ranked[:top_n]


class BM25Baseline:
    """Embed nothing -- lexical BM25 over train issue text, aggregate the resolvers of the
    top-N most similar train issues, weighted by BM25 score."""

    name = "bm25_knn"

    def __init__(self, k_neighbors: int = 50):
        self.k_neighbors = k_neighbors

    def fit(self, train_labeled):
        self.rows = train_labeled.reset_index(drop=True)
        corpus = [
            tokenize(f"{r.summary or ''} {r.description or ''}")
            for r in self.rows.itertuples()
        ]
        self.bm25 = BM25Okapi(corpus)
        self.assignees = self.rows["assignee_key"].tolist()
        self.fallback = PopularityBaseline()
        self.fallback.fit(train_labeled)

    def rank(self, issue_row, top_n=50):
        query = tokenize(f"{issue_row.get('summary') or ''} {issue_row.get('description') or ''}")
        if not query:
            return self.fallback.rank(issue_row, top_n)
        scores = self.bm25.get_scores(query)
        # top-k_neighbors most similar train issues by BM25 score
        top_idx = scores.argsort()[::-1][: self.k_neighbors]
        agg = Counter()
        for i in top_idx:
            if scores[i] <= 0:
                continue
            agg[self.assignees[i]] += float(scores[i])
        if not agg:
            return self.fallback.rank(issue_row, top_n)
        ranked = [d for d, _ in agg.most_common()]
        for d in self.fallback.ranking:
            if d not in ranked:
                ranked.append(d)
            if len(ranked) >= top_n:
                break
        return ranked[:top_n]
