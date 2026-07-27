# Phase 6 — Serving (minimal)

A working end-to-end demo, not the full HANDOFF Phase 6 spec (no p50/p99 latency table, no
dashboard) -- see `docs/decisions.md` for why this was scoped down for the MVP pass.

## What's here

- `scripts/08_train_and_save_artifacts.py` -- trains the same Phase 3 pipeline once per
  project and persists artifacts to `data/models/<PROJECT>.{faiss,pkl,_lgbm.txt}` (FAISS
  index, feature-builder state, LightGBM model) so the API doesn't retrain at startup.
- `src/issue_assignee_recommender/api.py` -- FastAPI app. Loads the shared sentence-transformer
  encoder once (identical across projects) plus all 5 projects' artifacts at startup (~10s
  cold start, vs. 3-8 minutes if it retrained per project).
- Endpoints:
  - `GET /health` -- which projects loaded
  - `GET /projects` -- Phase 3 leaderboard metrics per project
  - `POST /recommend/{project}` -- body: `{summary, description, components, reporter_key,
    top_n}` (only fields legitimately available at issue-creation time, per
    `docs/phase0_framing.md`); returns ranked `{developer_id, score}` candidates.

## Running it

```bash
.venv/bin/python scripts/08_train_and_save_artifacts.py   # once, ~20 min for all 5 projects
.venv/bin/uvicorn issue_assignee_recommender.api:app --app-dir src --host 127.0.0.1 --port 8000
```

## Smoke test (real request against real trained model)

```
POST /recommend/KAFKA
{"summary": "Consumer group rebalance causes duplicate message processing",
 "description": "When a consumer in the group restarts, we see duplicate processing of
   messages around the rebalance window. This looks related to offset commit timing.",
 "components": ["consumer"], "top_n": 5}
```
returned 5 ranked developer IDs with model scores in ~500ms (dominated by GPU embedding encode
on a cold single-request path -- batching or caching the encoder call would cut this further,
not done here).

## Known gaps vs. the full HANDOFF Phase 6 spec

- No p50/p99 latency table under a stated budget (single manual smoke test only).
- No dashboard / DuckDB experiment-tracking reuse from the LDPC project pattern.
- Single-request encoding path only -- no batching, no caching of repeated encoder calls.
- Developer IDs are returned as the dataset's raw anonymized `<<|author_key|UUID|>>` strings,
  not resolved to anything human-readable (there's nothing to resolve them to -- the dataset
  intentionally strips real identities).

None of these block the demo from working; they're the next things to do if this gets picked
back up for the full Phase 6 deliverable.
