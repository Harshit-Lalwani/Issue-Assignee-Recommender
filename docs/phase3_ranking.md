# Phase 3 — Learned Ranking

Per HANDOFF: "This is where the project becomes resume-worthy. Everything after is upside."

## Model

LightGBM `LGBMRanker` (`objective="lambdarank"`), 200 trees, over candidate sets generated per
issue from five sources (see "Candidate generation" below).

Features (`src/issue_assignee_recommender/features.py`), all computed from a frozen pre-cutoff
snapshot:

| feature | what it is |
|---|---|
| `dev_project_total` | lifetime resolution volume in this project |
| `dev_component_count` | resolution volume in this issue's component(s) |
| `dev_component_share` | that count as a share of all resolutions in those components |
| `dev_component_recent_365d` | component volume restricted to the trailing year |
| `dev_issuetype_count` | resolution volume for this issue's issuetype |
| `dev_reporter_affinity` | how often this developer resolved issues filed by this reporter |
| `dev_reporter_affinity_share` | that count as a share of the reporter's issues |
| `dev_recency_days` | days since the developer's last resolution, as of the cutoff |
| `dev_open_workload` | issues assigned but still open as of the cutoff |
| `dev_recent_volume_365d` / `_90d` | resolutions in the trailing year / quarter |
| `text_affinity_score` / `_rank` | embedding-retrieval similarity, and rank by it |
| `bm25_affinity_score` / `_rank` | BM25 lexical similarity, and rank by it |
| `is_known_dev` | whether the developer appears in the history snapshot at all |

The `_share` and `_rank` variants exist because the raw counts and scores are not comparable
across issues — an issue with a long description matches more neighbours and gets larger
affinity scores, and a component with 3,000 historical issues produces larger counts than one
with 30. Ranks and shares are scale-free, so a single tree threshold means the same thing on
every issue.

## Where the headroom is (measured, 2026-07-26)

The pipeline is two-stage, so `NDCG@10 = P(true assignee is in the candidate set) x (how well
the ranker orders it once it's there)`. Measuring those two factors separately is what
determined everything below.

With the original candidate generation (embedding top-50 neighbours' resolvers + top-20
component owners + top-20 globally popular, ~40-50 candidates):

| project | oracle recall (truth in candidate set) | ranker Recall@10 | share of ceiling captured |
|---|---|---|---|
| JRASERVER | 0.268 | 0.244 | 91% |
| SPARK | 0.512 | 0.493 | 96% |
| HADOOP | 0.350 | 0.328 | 94% |
| KAFKA | 0.507 | 0.476 | 94% |
| CASSANDRA | 0.349 | 0.289 | 83% |

**The ranker was already recovering 83-96% of whatever it was handed, inside 10 slots.** Nearly
all remaining error was issues where the true assignee was never a candidate at all — which no
amount of feature engineering, hyperparameter tuning, or LLM re-ranking can fix. This
retroactively explains two earlier null results: the Phase 5 LLM re-ranking experiment (which
could only reorder a list that already contained the answer 90%+ of the time) and the
`dev_recent_volume_365d` addition (`docs/decisions.md`, 2026-07-26).

Not all of the gap is recoverable. 18-35% of test issues are resolved by someone who resolved
*nothing* before the cutoff — invisible to any history-based system, so the achievable ceiling
is `1 - cold_start_fraction`.

## Candidate generation

Five sources, widened (`features.py` defaults: embeddings 200, BM25 100, component 50,
popularity 50, recently-active 50), giving ~110-170 candidates per issue:

| project | old (emb+comp+pop) | + wider cuts | + BM25 | + recently-active | achievable ceiling |
|---|---|---|---|---|---|
| JRASERVER | 0.268 | 0.439 | 0.496 | **0.597** | 0.723 |
| SPARK | 0.512 | 0.653 | 0.671 | **0.713** | 0.815 |
| HADOOP | 0.350 | 0.441 | 0.466 | **0.506** | 0.651 |
| KAFKA | 0.507 | 0.652 | 0.663 | **0.743** | 0.815 |
| CASSANDRA | 0.349 | 0.455 | 0.483 | **0.575** | 0.738 |

This lands within 78-91% of the achievable ceiling. Two sources were tested and **rejected**:

- **Commenters on retrieved similar issues**: +0.3-0.7 points of oracle recall for a ~40%
  larger candidate set. Not worth the cost.
- **Comments as a cold-start fix**: only 6-13% of cold-start true assignees had commented on
  anything before the cutoff, so comment history does not meaningfully shrink the unreachable
  slice either.

BM25 as a *candidate source* (rather than only a Phase 1 baseline) needs a query cost
compatible with running it on every issue. `rank_bm25.BM25Okapi` scores 41.5ms/query on
KAFKA's 5,566 train issues; `src/issue_assignee_recommender/lexical.py` precomputes the same
BM25 weights into a sparse term-document matrix and scores in 0.3ms — 140x, same scoring
function.

## Leakage boundary (read this before trusting the numbers)

This is the highest-risk part of the whole project per HANDOFF, so the split is two-level:

- **Test issues** are scored by a `FeatureBuilder` fit on *all* of train (everything before
  each project's split cutoff `T`) — unchanged from Phases 1-2.
- **Training examples** are the later slice of train only: issues created after an *inner*
  cutoff `T2` = 80th percentile of train's own `created` column. Their features come from a
  *separate* `FeatureBuilder` fit only on the earlier slice (`created < T2`). Since `T2`
  precedes every training example's own creation time, no training example's features can see
  its own outcome or anything that happened after it.
- The true assignee is force-added to **training** candidate sets only, so every training group
  has at least one positive. This affects which *labels* the supervised model sees (normal in
  supervised learning), not which *features* it can use — and is never done for test candidates.
- Every added source and feature respects the same rule: the BM25 index is built over history
  text only, `recently-active` ranks come from pre-cutoff resolutions only, and the component /
  reporter / issuetype aggregates are all snapshot counts.

## Results

| project | Recall@5 | Recall@10 | MRR | NDCG@10 | oracle recall | NDCG@10 vs Phase 1 BM25 |
|---|---|---|---|---|---|---|
| JRASERVER | 0.173 | 0.317 | 0.116 | 0.153 | 0.603 | 0.115 -> 0.153 (+34%) |
| SPARK | 0.644 | 0.663 | 0.626 | 0.633 | 0.710 | 0.213 -> 0.633 (+198%) |
| HADOOP | 0.401 | 0.428 | 0.380 | 0.389 | 0.504 | 0.175 -> 0.389 (+122%) |
| KAFKA | 0.595 | 0.632 | 0.555 | 0.569 | 0.744 | 0.204 -> 0.569 (+180%) |
| CASSANDRA | 0.398 | 0.431 | 0.356 | 0.370 | 0.576 | 0.111 -> 0.370 (+234%) |

Full numbers + per-project feature importances: `data/processed/phase3_results.csv`. Run log:
`data/processed/phase3_run_v3.log`.

### Attribution (ablation)

`scripts/07_phase3_ranking.py --legacy-candidates` reruns with the new feature set but the
original candidate widths, isolating the two changes:

| project | Phase 1 BM25 | v1 (narrow cands, 8 feats) | new feats, narrow cands | new feats, wide cands |
|---|---|---|---|---|
| JRASERVER | 0.115 | 0.126 | 0.128 | **0.153** |
| SPARK | 0.213 | 0.464 | 0.471 | **0.633** |
| HADOOP | 0.175 | 0.290 | 0.302 | **0.389** |
| KAFKA | 0.204 | 0.404 | 0.415 | **0.569** |
| CASSANDRA | 0.111 | 0.229 | 0.251 | **0.370** |

**The seven added features are worth +1 to +2 points of NDCG@10 (4-7% relative). Candidate
generation is worth the other ~85-90% of the gain.** Superseded v1 numbers are kept at
`data/processed/phase3_results_v1_narrow_candidates.csv`; the ablation at
`phase3_results_legacy_candidates.csv`.

## Reading these numbers

- **Substantial lift on all 5 projects**, but the honest framing is that most of it came from
  fixing a first-stage recall bug, not from better modeling. The ranker was never the weak part.
- **No leakage red flags.** The strongest check is that the ranker's *conversion rate* — NDCG@10
  divided by oracle recall — barely moved: SPARK was converting 91% of its candidate ceiling
  before and 89% after, KAFKA 79% -> 77%. If the new features were leaking, conversion would
  have jumped, not held flat while the ceiling rose. Feature importance is spread across all 15
  real features with no dominator, and `is_known_dev` still carries ~zero importance (correctly
  redundant with the count features).
- **JRASERVER has flipped from retrieval-limited to ranking-limited.** Its oracle recall more
  than doubled (0.268 -> 0.603) but NDCG@10 only went 0.126 -> 0.153: it now converts just 25%
  of its ceiling, against 64-89% everywhere else. This is a sharper problem statement than the
  earlier "sparse labels" reading — the candidates are there now and the behavioral features
  can't tell them apart, consistent with a 20-year tracker where past resolution history is a
  poor guide to who is active today. That, not more candidate sources, is where JRASERVER's
  next gain is.
- **Remaining headroom is now split.** Oracle recall still sits 12-15 points below the
  achievable ceiling on every project, and the cold-start slice (18-35% of test issues) is
  unreachable by construction — it is the single largest bucket of remaining error and needs a
  different signal entirely (e.g. non-resolution activity), not a better ranker.
