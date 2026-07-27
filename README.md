# Issue-Assignee Recommender

Given a newly-filed Jira issue, rank the developers most likely to resolve it — a two-stage
recommender system (retrieval + learned ranking) built and evaluated on real, public Jira data.

## Problem

Jira issues need an owner. This project treats "who should this be assigned to" as a
recommendation problem: developers are users, issues are items, and the ground truth (who
actually closed each issue) is unambiguous and requires no labeling.

## Architecture

1. **Retrieval** — narrow the full developer pool down to a shortlist of ~50 candidates using
   text similarity (BM25 / sentence embeddings) over historical issues, aggregating the
   developers who resolved similar issues in the past.
2. **Ranking** — reorder that shortlist with a learned model (LightGBM LambdaRank) over
   behavioral features: component ownership history, recency, current workload, and
   reporter/assignee affinity.
3. **Cold-start re-ranking** — for issues where behavioral history is weak (new contributors,
   new components), an LLM re-ranks the top candidates using retrieved historical context.
   Applied only to that slice, and only if it earns its keep on the numbers.

All temporal splits are strict: features and candidates are computed only from information
available at issue-creation time. Anything derived from resolution comments, close dates, or
other post-creation fields is leakage and is excluded by construction.

## Data

[The Public Jira Dataset](https://zenodo.org/records/15719919) (Montgomery, Lüders, Maalej —
MSR 2022): 16 public Jira instances (including Apache), anonymized, ~2.7M issues. This project
scopes to a handful of projects with genuinely distributed contribution — a project dominated
by one or two maintainers makes "assign it to whoever's busiest" unbeatable and the exercise
uninformative.

## Status

Phases 0-3 and a minimal Phase 6 are done and measured end to end, on 5 real projects
(JRASERVER — Jira's own tracker — plus SPARK, HADOOP, KAFKA, CASSANDRA):

| phase | deliverable | headline result |
|---|---|---|
| [0](docs/phase0_framing.md) | data + task framing | 129k issues exported, frozen per-project temporal splits |
| [1](docs/phase1_baselines.md) | baselines + eval harness | BM25 kNN beats popularity/component-owner on every project |
| [2](docs/phase2_retrieval.md) | semantic retrieval | embeddings ~= BM25 on Recall@50 — complementary, not interchangeable (Phase 4 gate stays closed) |
| [3](docs/phase3_ranking.md) | learned ranking | LightGBM lifts NDCG@10 34-234% over the best baseline on all 5 projects |
| 5 | LLM cold-start re-ranking | benchmarked against 4 providers, made rankings *worse* — closed as a negative result, not shipped (see `docs/decisions.md`) |
| [6](docs/phase6_serving.md) | serving (minimal) | FastAPI endpoint, real trained model, ~40ms/request |

Current Phase 3 test-set numbers, and the candidate-set ceiling each is working against:

| project | Recall@5 | NDCG@10 | oracle recall | unreachable (cold start) |
|---|---|---|---|---|
| JRASERVER | 0.173 | 0.153 | 0.603 | 28% |
| SPARK | 0.644 | 0.633 | 0.710 | 19% |
| HADOOP | 0.401 | 0.389 | 0.504 | 35% |
| KAFKA | 0.595 | 0.569 | 0.744 | 18% |
| CASSANDRA | 0.398 | 0.370 | 0.576 | 26% |

A full, plain-language walkthrough of the problem, the data, and every result above (written
for someone with no prior context on Jira, recommender systems, or the ML tooling used) is in
[`walkthrough.md`](walkthrough.md). For the reasoning behind every non-obvious judgment call —
project selection, split methodology, the Phase 4/5 gate decisions, the two negative feature
results — see [`docs/decisions.md`](docs/decisions.md).

## Layout

```
src/issue_assignee_recommender/   # library code (data, features, baselines, retrieval, llm, api)
scripts/                          # numbered pipeline scripts, run in order (01-09)
notebooks/                        # exploration
tests/                            # unit tests
data/                             # raw dump + trained model artifacts (gitignored); small
                                   # measured-results CSVs under data/processed/ are tracked
docs/                             # phase deliverables, decisions log, design notes
walkthrough.md                    # full plain-language explanation of the project and results
```

## Running

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Pipeline scripts run in numeric order (`scripts/01_...` through `09_...`) — see each phase's
doc in `docs/` for what it produces and how to reproduce it. `09_phase5_benchmark_models.py` is
the closed, not-shipped Phase 5 experiment; it needs real API keys in `.env` (see
`.env.example`) and isn't part of the serving path. To bring up the API once artifacts are
trained (`scripts/08_train_and_save_artifacts.py`):

```bash
.venv/bin/uvicorn issue_assignee_recommender.api:app --app-dir src --host 127.0.0.1 --port 8000
```
