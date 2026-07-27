# Deferred decisions

Running log of choices made unilaterally overnight to keep momentum toward an MVP, plus
anything postponed rather than decided. Each entry: what was decided/deferred, why, and what
to revisit.

Convention: entries append at the bottom with a timestamp (UTC). Nothing here is final —
review and override anything you disagree with.

---

## 2026-07-25 18:30 UTC — Working mode for the night

**Decision:** Where the HANDOFF leaves a judgment call and no answer blocks correctness or
leakage-safety, pick whatever gets an end-to-end MVP (Phase 0 -> Phase 3, then a minimal Phase
6 serving demo) working fastest, log it here, and keep moving rather than stopping to ask.
Gated phases (4: fine-tuning, 5: LLM re-ranking) stay gated per HANDOFF — not started unless
Phase 2/3 results explicitly show headroom, per the original brief.

**Why:** User is offline for the night and explicitly asked for autonomous forward progress
over correctness debates on non-critical calls.

**Revisit:** Every decision below is independently revisitable; this entry just explains why
they were made without a pause.

---

## 2026-07-25 19:00 UTC — Project selection (Phase 0)

**Decision:** Selected 5 projects via a Mongo aggregation over all 16 instances, filtered to
issue_count >= 2000, distinct_resolvers >= 20, top2-resolver-share <= 0.35, then picked for a
mix of scale and narrative:

| project_key | instance | issues | distinct_resolvers | top2_share | date range |
|---|---|---|---|---|---|
| JRASERVER | Jira | 47,225 | 484 | 0.123 | 2002-2022 |
| SPARK | Apache | 37,443 | 1,783 | 0.068 | 2010-2022 (1 anomalous pre-2010 timestamp filtered) |
| HADOOP | Apache | 15,797 | 995 | 0.082 | 2005-2022 |
| KAFKA | Apache | 12,312 | 601 | 0.101 | 2011-2022 |
| CASSANDRA | Apache | 17,115 | 698 | 0.162 | 2009-2022 |

**Why:** All 5 clear the "not dominated by 1-2 maintainers" bar the HANDOFF calls out as
existential (top2_share all <=0.16, well under the 0.5 danger line). JRASERVER is Jira's own
issue tracker — as on-domain as this project can get for an Atlassian application ("rebuilds a
piece of their product" per HANDOFF). SPARK/HADOOP/KAFKA/CASSANDRA are recognizable, large,
well-distributed, and span genuinely different technical vocabularies (batch processing,
distributed FS, streaming, distributed DB), which stresses the text-retrieval stage more
usefully than 5 modules of the same monorepo would.

**Revisit:** Could add HBASE/HIVE/FLINK/YARN/AMBARI/IGNITE later — all passed the same filter
and are sitting in `data/processed/project_selection_stats.csv` (469 rows total, all projects
>=500 issues, most passing the concentration filter too). Nothing about the pipeline is
specific to these 5; swapping is a config change, not a rewrite.

## 2026-07-25 19:00 UTC — Temporal split methodology

**Decision:** Per-project split, not one global cutoff date. Each project's train/test cutoff
T is set at its own 80th percentile of `created` timestamp (so ~80% train / ~20% test issues
per project, chronologically). Also decided to model these as 5 **independent per-project
recommenders** rather than one pooled model, per HANDOFF's note that "per-project models are
arguably the more realistic production setup anyway."

**Why:** A single global date would give wildly different train/test ratios across projects
with different lifespans and issue velocity (JRASERVER has 20 years of history, KAFKA ~11) --
percentile-based keeps every project's test set a comparable, meaningfully-sized slice
(~2,500-9,400 issues depending on project) without hand-tuning five dates.

**Revisit:** If cross-project pooling turns out to help cold-start (Phase 5) via shared
vocabulary, that's a legitimate reason to revisit and build one pooled retrieval index over
per-project ranking -- not decided against, just not the Phase 0-3 default.

---

## 2026-07-26 12:30 UTC — Phase 4 (fine-tuning) gate: not triggered

**Decision:** Not doing Phase 4. Off-the-shelf `all-MiniLM-L6-v2` embeddings scored Recall@50
within +/-2 points of BM25 on 4/5 projects and -9 points on JRASERVER (see
`docs/phase2_retrieval.md`) -- no headroom for a fine-tune to close.

**Why:** HANDOFF gates Phase 4 explicitly on Phase 2 showing headroom over the off-the-shelf
encoder, and states the not-fine-tuning call is the better engineering call when embeddings are
already near-ceiling. They're not even beating the non-ML baseline here, so fine-tuning would
be optimizing the wrong stage of the pipeline.

**Revisit:** If Phase 3 ranking results show the embedding-similarity feature is doing real
work despite the retrieval-stage parity (i.e. it's not redundant with BM25/behavioral
features), that would be a signal worth a fine-tuning pass later -- not before.

Update 2026-07-26: it is doing real work -- `text_affinity_score` (the embedding similarity
feature) is the single strongest feature in every Phase 3 model, ~30-35% of total importance.
Doesn't reopen the Phase 4 gate on its own (the retrieval-stage Recall@50 parity with BM25
still holds, and the feature helping the ranker isn't the same claim as "a fine-tune would
improve it further") but worth knowing if Phase 4 comes up again.

---

## 2026-07-26 13:30 UTC — Phase 5 (LLM re-ranking) gate: unlocked, deferred

**Decision:** Phase 5 is gated on "Phase 3 works" -- it does (see `docs/phase3_ranking.md`:
66-118% NDCG@10 lift over the best Phase 1 baseline on 4/5 projects). Not starting it in this
session anyway.

**Why:** Phase 5 needs LLM API access and incurs real per-call cost (HANDOFF estimates
$2-10 in tokens for the cold-start slice). No API key/budget was set up for this autonomous
run, and spending money without the user present to confirm isn't a call to make
unilaterally -- unlike the modeling decisions above, this one has a real-world cost attached.

**Revisit:** Gate is open. Next step whenever picked back up: identify the cold-start slice
per project (new contributors / new components / sparse history -- not yet defined), get API
access confirmed, then re-rank top-10 candidates on that slice only and measure lift + latency
+ cost, per HANDOFF's Phase 5 spec. A negative result is an acceptable, reportable outcome.

**Update 2026-07-26 20:15 UTC — concluded, not shipping.** Benchmarked for real (see
`docs/decisions.md`'s "multi-provider LLM client" entry below and
`data/processed/phase5_benchmark{,2}.log`): on 21 real cold-start-like test issues, LLM
re-ranking of the LightGBM top-10 made results *worse*, not better, with every model tested and
both a stats-only and a retrieved-context-enriched prompt (best case: Recall@5 dropped from
0.905 with LightGBM alone to 0.524 after re-ranking). Per explicit user instruction, not
building this into the serving pipeline. Phase 5 is closed as a genuine, measured negative
result -- the multi-provider LLM client (`src/issue_assignee_recommender/llm.py`) stays in the
repo as the artifact of that experiment, but `api.py` does not call it and won't by default.

---

## 2026-07-26 13:30 UTC — Phase 6 (serving): building a minimal version now

**Decision:** Building a minimal FastAPI serving endpoint next, using the already-trained
Phase 3 artifacts (LightGBM model + embedding index per project), rather than treating Phase 6
as fully out of scope for this session.

**Why:** MVP-first working mode (see top entry) -- an end-to-end demo (issue in, ranked
developers out) is worth more right now than polishing Phases 0-3 further, and all the pieces
it needs already exist. Latency/dashboard polish from the full HANDOFF Phase 6 spec (DuckDB
experiment tracking pattern, p50/p99 tables) is left for a follow-up pass, not done here.

**Revisit:** Whether to persist trained models to disk (pickle/joblib) vs retrain at server
startup -- going with retrain-at-startup for now for simplicity (training is ~1-6 min per
project per Phase 3 timings), revisit if that latency becomes annoying in practice.

---

## 2026-07-26 20:00 UTC — Added trailing-window recency feature; it didn't help

**Decision:** Added `dev_recent_volume_365d` (count of a developer's resolutions in the
trailing 365 days as of the feature cutoff) to the Phase 3 feature set, as a genuine gap fix,
not a hypothetical -- kept in the feature set despite the result below since it's harmless and
more conceptually correct than lifetime-only counts.

**Why:** User asked directly why timestamp-windowed features (e.g. "issues resolved in the
past year") weren't already part of the model, specifically as a fix for JRASERVER's flagged
20-year sparse-turnover problem. `dev_project_total` was a lifetime count with no decay --
someone active only a decade ago looked identical to someone steadily active now. Real gap,
worth testing rather than just acknowledging.

**Result: negligible effect everywhere, including JRASERVER** (the project this was meant to
help). NDCG@10 before/after: JRASERVER 0.126->0.126, SPARK 0.465->0.464, HADOOP 0.291->0.290,
KAFKA 0.400->0.404, CASSANDRA 0.234->0.229 -- noise-level movement in both directions, not a
real lift anywhere. `dev_recent_volume_365d` gets non-zero but consistently the smallest
feature importance of the six real features (124-386 vs. 653-2430 for the others). Most likely
explanation: LightGBM's trees were already approximating "recently active vs. not" by combining
`dev_recency_days` and `dev_project_total`, so the explicit window is largely redundant
information, not new information.

**Revisit:** JRASERVER's weak numbers more likely trace to what's already documented in
`docs/phase0_framing.md` -- only ~25% of its issues ever get both an assignee and a resolution,
so there's less labeled training signal per candidate regardless of feature representation.
That's a labeled-data problem, not a feature-engineering problem, and no feature addition is
likely to fix it. If revisited, a different angle (e.g. treating "still open / never resolved"
issues as a distinct signal, or a different window length) would be the next thing to try, not
another variant of trailing-volume.

---

## 2026-07-26 21:30 UTC — Candidate generation was the bottleneck, not the ranker

**Decision:** Widened Phase 3 candidate generation from 3 sources (~45 candidates) to 5
(~110-170): embedding neighbours 50->200, component owners 20->50, popularity 20->50, plus two
new sources — BM25 top-100 neighbours' resolvers, and the top-50 developers by trailing-365d
resolution volume. Also added 7 features (BM25 affinity score/rank, embedding rank, component
and reporter *shares*, component-recent-365d, issuetype affinity, 90d volume).

**Why:** Measured the two-stage decomposition properly for the first time — `NDCG@10 =
P(truth in candidate set) x ranking quality given that`. The ranker was already recovering
83-96% of its candidate ceiling within 10 slots, so essentially all remaining error was issues
whose true assignee was never a candidate. Every previous improvement attempt had been aimed at
the stage that was already working.

**Result: NDCG@10 up 21-61% relative** (JRASERVER 0.126->0.153, SPARK 0.464->0.633, HADOOP
0.290->0.389, KAFKA 0.404->0.569, CASSANDRA 0.229->0.370). An ablation
(`07_phase3_ranking.py --legacy-candidates`, results in
`data/processed/phase3_results_legacy_candidates.csv`) attributes +1-2 points to the new
features and the remaining ~85-90% of the gain to candidate generation. Cost: ~3x candidates
per issue, but net *faster* than before, because `FeatureBuilder.score_issue()` replaced the old
`candidates()` + `retrieval_scores()` pair that re-encoded and re-searched every issue twice.

**Rejected after measuring:** commenters on retrieved similar issues as a 6th source (+0.3-0.7
points of oracle recall for a ~40% bigger candidate set — not worth it).

**Revisit:** Oracle recall is still 12-15 points below the achievable ceiling on every project.
The bigger remaining bucket is cold start: 18-35% of test issues are resolved by someone with
zero pre-cutoff resolutions, and only 6-13% of those people even commented pre-cutoff, so
comment history won't fix it. That slice needs a different signal (or an explicit "no confident
recommendation" abstention), not more retrieval tuning.

## 2026-07-26 21:30 UTC — Retracting the Phase 2 latency conclusion

**Decision:** Withdrew the "embedding retrieval is 15-50x faster than BM25" conclusion from
`docs/phase2_retrieval.md`, and added `src/issue_assignee_recommender/lexical.py` (sparse BM25).

**Why:** That measurement compared FAISS against `rank_bm25.BM25Okapi`, whose per-query cost is
a full pure-Python pass over the corpus. Precomputing the identical BM25 weights into a sparse
term-document matrix gives the same scores at 0.3ms/query vs 41.5ms (KAFKA, 5,566 docs) — 140x.
The speed gap measured the library, not the method. This also made BM25 cheap enough to use as a
per-issue candidate source, which is what unlocked the gains in the entry above.

**Also revised:** the Phase 2 read that embeddings and BM25 are "interchangeable, pick one"
because their Recall@50 was tied. They tie in *rate* but miss on different issues — using both
as candidate sources beats either alone, and both affinity features rank at the top of every
Phase 3 model. Phase 1 leaderboard numbers are untouched (still `BM25Okapi`, frozen).

**Revisit:** The Phase 4 (fine-tuning) gate stays closed — nothing here says the *encoder* has
headroom. If anything the evidence points further away from encoder work: the win came from
retrieving more broadly, not from matching better.

## 2026-07-26 21:30 UTC — JRASERVER reclassified: ranking-limited, not label-limited

**Decision:** Superseded the earlier read (2026-07-26, trailing-window feature entry) that
JRASERVER's weak numbers are a labeled-data problem nothing can fix.

**Why:** With wider candidate generation its oracle recall more than doubled (0.268 -> 0.603),
but NDCG@10 moved only 0.126 -> 0.153. It now converts 25% of its candidate ceiling versus
64-89% on every other project. The candidates *are* there; the behavioral features can't
separate them. That is a ranking/feature problem specific to a 20-year tracker with high
turnover, not an absence of training signal.

**Revisit:** The next thing to try for JRASERVER is time-decayed rather than snapshot-count
features (exponential decay on resolution history, or restricting the whole feature snapshot to
a trailing window instead of all history) — note this is *not* the same as the trailing-volume
feature already tested, which added a window feature alongside unchanged lifetime counts rather
than replacing the snapshot.
