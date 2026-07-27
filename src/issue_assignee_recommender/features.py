"""Phase 3 feature engineering.

FeatureBuilder is fit on a "history" snapshot (all activity strictly before some cutoff
timestamp) and can then score arbitrary (issue, candidate developer) pairs using only that
snapshot. This is the leakage boundary: callers must guarantee every issue they score with a
given FeatureBuilder was created at or after that builder's cutoff -- see
docs/phase3_ranking.md for how train vs test snapshots are kept separate.

Candidate generation is the measured bottleneck of the whole pipeline (docs/phase3_ranking.md,
"Where the headroom is"): the ranker recovers 83-96% of whatever its candidate set contains, so
the numbers move by putting the true assignee in that set more often, not by ranking harder.
Hence five candidate sources rather than three, and wider cuts of each.
"""
import pickle
from collections import Counter, defaultdict
from datetime import timedelta

import faiss

from .baselines import _as_list
from .lexical import SparseBM25
from .retrieval import EmbeddingRetriever
from .text import tokenize

# Candidate-source widths. Defaults chosen from the sweep in docs/phase3_ranking.md: this
# configuration roughly doubles oracle recall vs. the original (50/20/20, embeddings only) at
# ~3x the candidate-set size, and lands within 78-91% of the achievable ceiling (i.e. of the
# share of test issues whose true assignee resolved anything at all before the cutoff).
DEFAULT_EMB_K = 200
DEFAULT_BM25_K = 100
DEFAULT_COMP_TOP = 50
DEFAULT_POP_TOP = 50
DEFAULT_RECENT_TOP = 50

MISSING_RANK = 999.0


class FeatureBuilder:
    def __init__(
        self,
        model,
        cutoff,
        k_neighbors: int = DEFAULT_EMB_K,
        bm25_k: int = DEFAULT_BM25_K,
        comp_top: int = DEFAULT_COMP_TOP,
        pop_top: int = DEFAULT_POP_TOP,
        recent_top: int = DEFAULT_RECENT_TOP,
    ):
        self.cutoff = cutoff
        self.bm25_k = bm25_k
        self.comp_top = comp_top
        self.pop_top = pop_top
        self.recent_top = recent_top
        self.retriever = EmbeddingRetriever(model, k_neighbors=k_neighbors)

    def fit(self, history_labeled, history_all):
        """history_labeled: resolved+assigned issues strictly before cutoff (signal source).
        history_all: ALL issues strictly before cutoff, labeled or not (needed for open
        workload -- an issue can be currently assigned with no resolution yet)."""
        self.retriever.fit(history_labeled)

        texts = [
            f"{r.summary or ''} {r.description or ''}" for r in self.retriever.rows.itertuples()
        ]
        self.bm25 = SparseBM25([tokenize(t) for t in texts])

        self.popularity = Counter(history_labeled["assignee_key"])

        recent_window_start = self.cutoff - timedelta(days=365)
        recent_window_start_90 = self.cutoff - timedelta(days=90)

        self.component_counts: dict[str, Counter] = defaultdict(Counter)
        self.component_recent: dict[str, Counter] = defaultdict(Counter)
        self.component_totals: Counter = Counter()
        self.reporter_affinity: dict[str, Counter] = defaultdict(Counter)
        self.reporter_totals: Counter = Counter()
        self.issuetype_counts: dict[str, Counter] = defaultdict(Counter)
        last_activity: dict[str, object] = {}
        recent_volume: Counter = Counter()
        recent_volume_90: Counter = Counter()

        for row in history_labeled.itertuples():
            dev = row.assignee_key
            for comp in _as_list(row.components):
                self.component_counts[comp][dev] += 1
                self.component_totals[comp] += 1
                if row.created >= recent_window_start:
                    self.component_recent[comp][dev] += 1
            if row.reporter_key:
                self.reporter_affinity[row.reporter_key][dev] += 1
                self.reporter_totals[row.reporter_key] += 1
            if getattr(row, "issuetype", None):
                self.issuetype_counts[row.issuetype][dev] += 1
            if dev not in last_activity or row.created > last_activity[dev]:
                last_activity[dev] = row.created
            # Trailing-window volume, separate from lifetime `popularity`: someone who resolved
            # 500 issues a decade ago and nothing since should not look identical to someone
            # doing steady recent work.
            if row.created >= recent_window_start:
                recent_volume[dev] += 1
            if row.created >= recent_window_start_90:
                recent_volume_90[dev] += 1

        self.last_activity = last_activity
        self.recent_volume = recent_volume
        self.recent_volume_90 = recent_volume_90

        workload = Counter()
        for row in history_all.itertuples():
            dev = row.assignee_key
            if not dev:
                continue
            still_open = row.resolutiondate is None or (
                row.resolutiondate is not None and row.resolutiondate > self.cutoff
            )
            if still_open:
                workload[dev] += 1
        self.open_workload = workload

        self.global_ranking = [d for d, _ in self.popularity.most_common()]
        self.recent_ranking = [d for d, _ in recent_volume.most_common()]

    # ------------------------------------------------------------------ scoring

    def score_issue(self, issue_row: dict) -> dict:
        """One pass over every retrieval source for a single issue: returns the candidate set
        plus the per-source affinity scores/ranks the feature rows need. Callers should use
        this rather than calling `candidates()` and `retrieval_scores()` separately -- those
        each re-encode and re-search the same text."""
        text = f"{issue_row.get('summary') or ''} {issue_row.get('description') or ''}"
        cand: set = set()

        emb_scores: dict[str, float] = defaultdict(float)
        bm25_scores: dict[str, float] = defaultdict(float)

        if text.strip():
            qvec = self.retriever.encode_query(text)
            scores, idx = self.retriever.index.search(qvec, self.retriever.k_neighbors)
            for score, i in zip(scores[0], idx[0]):
                if i < 0:
                    continue
                dev = self.retriever.assignees[i]
                cand.add(dev)
                if score > 0:
                    emb_scores[dev] += float(score)
            for i, score in self.bm25.search(text, self.bm25_k):
                dev = self.retriever.assignees[i]
                cand.add(dev)
                bm25_scores[dev] += score

        for comp in _as_list(issue_row.get("components")):
            for dev, _ in self.component_counts.get(comp, Counter()).most_common(self.comp_top):
                cand.add(dev)
        cand.update(self.global_ranking[: self.pop_top])
        cand.update(self.recent_ranking[: self.recent_top])
        cand.discard(None)

        return {
            "candidates": cand,
            "emb_scores": dict(emb_scores),
            "bm25_scores": dict(bm25_scores),
            "emb_ranks": _ranks(emb_scores),
            "bm25_ranks": _ranks(bm25_scores),
        }

    def candidates(self, issue_row: dict) -> set:
        return self.score_issue(issue_row)["candidates"]

    def retrieval_scores(self, issue_row: dict) -> dict:
        return self.score_issue(issue_row)["emb_scores"]

    def feature_row(self, issue_row: dict, dev: str, ctx: dict) -> dict:
        """ctx is the dict returned by score_issue() for this issue."""
        components = _as_list(issue_row.get("components"))
        comp_count = sum(self.component_counts.get(c, {}).get(dev, 0) for c in components)
        comp_recent = sum(self.component_recent.get(c, {}).get(dev, 0) for c in components)
        comp_total = sum(self.component_totals.get(c, 0) for c in components)
        comp_share = comp_count / comp_total if comp_total else 0.0

        reporter = issue_row.get("reporter_key")
        rep_count = self.reporter_affinity.get(reporter, {}).get(dev, 0) if reporter else 0
        rep_total = self.reporter_totals.get(reporter, 0) if reporter else 0
        rep_share = rep_count / rep_total if rep_total else 0.0

        issuetype = issue_row.get("issuetype")
        type_count = self.issuetype_counts.get(issuetype, {}).get(dev, 0) if issuetype else 0

        last_seen = self.last_activity.get(dev)
        if last_seen is None:
            recency_days = 9999.0
        else:
            recency_days = max((self.cutoff - last_seen).total_seconds() / 86400.0, 0.0)

        return {
            "dev_project_total": self.popularity.get(dev, 0),
            "dev_component_count": comp_count,
            "dev_component_share": comp_share,
            "dev_component_recent_365d": comp_recent,
            "dev_issuetype_count": type_count,
            "dev_reporter_affinity": rep_count,
            "dev_reporter_affinity_share": rep_share,
            "dev_recency_days": recency_days,
            "dev_open_workload": self.open_workload.get(dev, 0),
            "dev_recent_volume_365d": self.recent_volume.get(dev, 0),
            "dev_recent_volume_90d": self.recent_volume_90.get(dev, 0),
            "text_affinity_score": ctx["emb_scores"].get(dev, 0.0),
            "text_affinity_rank": ctx["emb_ranks"].get(dev, MISSING_RANK),
            "bm25_affinity_score": ctx["bm25_scores"].get(dev, 0.0),
            "bm25_affinity_rank": ctx["bm25_ranks"].get(dev, MISSING_RANK),
            "is_known_dev": 1 if dev in self.popularity else 0,
        }

    # ------------------------------------------------------------------ persistence

    def save(self, path_prefix: str):
        """Persist everything except the shared SentenceTransformer encoder (passed back in
        at load time -- it's identical across all projects, no point duplicating it on disk)."""
        faiss.write_index(self.retriever.index, f"{path_prefix}.faiss")
        state = {
            "cutoff": self.cutoff,
            "k_neighbors": self.retriever.k_neighbors,
            "bm25_k": self.bm25_k,
            "comp_top": self.comp_top,
            "pop_top": self.pop_top,
            "recent_top": self.recent_top,
            "assignees": self.retriever.assignees,
            "bm25": self.bm25,
            "popularity": self.popularity,
            "component_counts": dict(self.component_counts),
            "component_recent": dict(self.component_recent),
            "component_totals": self.component_totals,
            "reporter_affinity": dict(self.reporter_affinity),
            "reporter_totals": self.reporter_totals,
            "issuetype_counts": dict(self.issuetype_counts),
            "last_activity": self.last_activity,
            "open_workload": self.open_workload,
            "recent_volume": self.recent_volume,
            "recent_volume_90": self.recent_volume_90,
            "global_ranking": self.global_ranking,
            "recent_ranking": self.recent_ranking,
        }
        with open(f"{path_prefix}.pkl", "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path_prefix: str, model):
        with open(f"{path_prefix}.pkl", "rb") as f:
            state = pickle.load(f)
        fb = cls(
            model,
            cutoff=state["cutoff"],
            k_neighbors=state["k_neighbors"],
            bm25_k=state["bm25_k"],
            comp_top=state["comp_top"],
            pop_top=state["pop_top"],
            recent_top=state["recent_top"],
        )
        fb.retriever.index = faiss.read_index(f"{path_prefix}.faiss")
        fb.retriever.assignees = state["assignees"]
        fb.bm25 = state["bm25"]
        fb.popularity = state["popularity"]
        fb.component_counts = defaultdict(Counter, state["component_counts"])
        fb.component_recent = defaultdict(Counter, state["component_recent"])
        fb.component_totals = state["component_totals"]
        fb.reporter_affinity = defaultdict(Counter, state["reporter_affinity"])
        fb.reporter_totals = state["reporter_totals"]
        fb.issuetype_counts = defaultdict(Counter, state["issuetype_counts"])
        fb.last_activity = state["last_activity"]
        fb.open_workload = state["open_workload"]
        fb.recent_volume = state["recent_volume"]
        fb.recent_volume_90 = state["recent_volume_90"]
        fb.global_ranking = state["global_ranking"]
        fb.recent_ranking = state["recent_ranking"]
        return fb


def _ranks(scores: dict) -> dict:
    """1-based rank of each developer by descending score -- a scale-free companion to the raw
    affinity scores, whose magnitude varies a lot between issues (long descriptions match more
    neighbours) and so is hard for the trees to threshold on directly."""
    return {
        dev: float(i + 1)
        for i, (dev, _) in enumerate(sorted(scores.items(), key=lambda kv: -kv[1]))
    }


FEATURE_COLUMNS = [
    "dev_project_total",
    "dev_component_count",
    "dev_component_share",
    "dev_component_recent_365d",
    "dev_issuetype_count",
    "dev_reporter_affinity",
    "dev_reporter_affinity_share",
    "dev_recency_days",
    "dev_open_workload",
    "dev_recent_volume_365d",
    "dev_recent_volume_90d",
    "text_affinity_score",
    "text_affinity_rank",
    "bm25_affinity_score",
    "bm25_affinity_rank",
    "is_known_dev",
]
