# Phase 1 — Evaluation Harness and Baselines

Per HANDOFF: this is the phase that's most important to get right before building anything
more sophisticated, because if nothing beats "assign it to whoever's busiest," no later result
means anything.

## Harness

`src/issue_assignee_recommender/metrics.py` — Recall@5, Recall@10, MRR, NDCG@10, computed per
issue against a single ground-truth assignee (the actual resolver), then averaged. Evaluated
strictly on each project's **labeled test subset** (issues with both `assignee_key` and
`resolutiondate` set) — see `docs/phase0_framing.md` for why this shrinks JRASERVER's usable
test set to 856 issues out of 9,445.

## Baselines (`src/issue_assignee_recommender/baselines.py`)

1. **Popularity** — rank developers by train-period resolution count, identical ranking for
   every issue.
2. **Component-owner** — rank by resolution count restricted to the issue's own component(s);
   falls back to popularity when an issue has no components or components unseen in train.
3. **BM25 kNN** — BM25 over train issue text (summary + description), retrieve the 50 most
   textually similar train issues per test issue, aggregate their resolvers weighted by BM25
   score.

All three `fit()` only on train-period issues that have a resolved assignee — consistent with
the leakage rule that "who resolved this" is itself outcome information, legitimately knowable
only for historical (train) issues.

## Results

| project | method | n | Recall@5 | Recall@10 | MRR | NDCG@10 |
|---|---|---|---|---|---|---|
| JRASERVER | popularity | 856 | 0.007 | 0.058 | 0.011 | 0.022 |
| JRASERVER | component_owner | 856 | 0.091 | 0.153 | 0.072 | 0.091 |
| JRASERVER | **bm25_knn** | 856 | **0.140** | **0.201** | 0.088 | 0.115 |
| SPARK | popularity | 4680 | 0.039 | 0.180 | 0.037 | 0.070 |
| SPARK | component_owner | 4680 | 0.149 | 0.200 | 0.085 | 0.112 |
| SPARK | **bm25_knn** | 4680 | **0.249** | **0.316** | 0.181 | 0.213 |
| HADOOP | popularity | 1924 | 0.137 | 0.146 | 0.069 | 0.089 |
| HADOOP | component_owner | 1924 | 0.176 | 0.231 | 0.152 | 0.170 |
| HADOOP | **bm25_knn** | 1924 | **0.200** | **0.241** | 0.155 | 0.175 |
| KAFKA | popularity | 1056 | 0.118 | 0.177 | 0.101 | 0.119 |
| KAFKA | component_owner | 1056 | 0.188 | 0.305 | 0.148 | 0.184 |
| KAFKA | **bm25_knn** | 1056 | **0.248** | **0.319** | 0.168 | 0.204 |
| CASSANDRA | popularity | 1921 | 0.129 | 0.156 | 0.035 | 0.063 |
| CASSANDRA | component_owner | 1921 | 0.113 | 0.177 | 0.056 | 0.084 |
| CASSANDRA | **bm25_knn** | 1921 | **0.137** | **0.206** | 0.082 | 0.111 |

Full leaderboard: `data/processed/phase1_leaderboard.csv`. Raw run log with per-method timing:
`data/processed/phase1_run.log`.

## Reading these numbers

- **BM25 kNN wins on every single project**, on every metric, over both popularity and
  component-owner. This is the load-bearing finding of Phase 1: the popularity baseline is
  genuinely beatable here (unlike many recsys tasks), which justifies building Phase 2/3 at
  all -- per the project selection filter in `docs/decisions.md`, none of these 5 projects are
  dominated by 1-2 maintainers, so "assign it to whoever's busiest" was never going to be
  strong.
- **JRASERVER's popularity baseline is unusually weak** (Recall@5 = 0.007, near-random for
  ~400 train resolvers). This is consistent with, not contradicted by, the sparse-label /
  long-history data-quality note in `docs/phase0_framing.md`: 14 years separate its 80th
  percentile split date (2016) from its earliest issues (2002), so the most active historical
  resolvers have plausibly moved on by the test period. Component-owner and BM25 partially
  recover from this by conditioning on issue content rather than raw historical volume.
- **Magnitudes are in the "modest but real" territory the HANDOFF flags as expected** for this
  task -- no Recall@5 near 1.0, no test-beats-train inversions, no single dominating feature
  (these are simple count/lexical models, that check doesn't fully apply yet but will in
  Phase 3). Nothing here reads as leakage.
- Component-owner beats popularity everywhere except CASSANDRA, where it's roughly tied/worse
  -- plausibly because CASSANDRA's `components` field is used less consistently by reporters
  than the other projects (worth a follow-up check in Phase 2/3, not blocking).

## Performance note

BM25 kNN is the slow baseline: pure-Python `rank_bm25` scores every test query against the
full train corpus with no index, so ~57 minutes total across all 5 projects (JRASERVER 509s,
SPARK 1874s, HADOOP 344s, KAFKA 136s, CASSANDRA 585s -- roughly scales with train-corpus size
x test-set size). This is exactly the cost the HANDOFF's two-stage architecture exists to avoid
at scale; Phase 2's FAISS flat index replaces this exhaustive score-everything approach with
approximate-free but genuinely indexed retrieval.
