# Phase 2 — Semantic Retrieval

## Setup

`src/issue_assignee_recommender/retrieval.py` — off-the-shelf `all-MiniLM-L6-v2` sentence
embeddings (384-dim), indexed per project with an exact `faiss.IndexFlatIP` over normalized
vectors (cosine similarity). At this scale (5k-18k train vectors per project, well under the
200k the HANDOFF sizes for) a flat index is fast enough that IVF/HNSW would only add tuning
surface for no measurable benefit — skipped per the HANDOFF's explicit guidance.

Retrieval methodology matches Phase 1's BM25 baseline exactly so the comparison isolates
text-similarity quality: retrieve the 50 nearest train issues, aggregate their resolvers
weighted by similarity score, pad to 50 candidates with global popularity if fewer than 50
distinct resolvers turn up.

## Results — Recall@50

| project | bm25_knn | embedding_faiss | delta |
|---|---|---|---|
| JRASERVER | 0.317 | 0.225 | **-0.091** |
| SPARK | 0.511 | 0.513 | +0.002 |
| HADOOP | 0.338 | 0.323 | -0.016 |
| KAFKA | 0.482 | 0.491 | +0.010 |
| CASSANDRA | 0.386 | 0.372 | -0.014 |

Full numbers: `data/processed/phase2_recall50.csv`. Run log with per-project timing:
`data/processed/phase2_run.log`.

## Reading these numbers

**Embeddings do not beat BM25 here.** They're statistically tied on SPARK and KAFKA (+/-1
point), slightly worse on HADOOP and CASSANDRA (-1 to -2 points), and clearly worse on
JRASERVER (-9 points). This is the HANDOFF's predicted "barely better" outcome, and the
predicted reason holds: these issue trackers are dense with identifiers, stack traces, class
names, and config keys (`NullPointerException`, `spark.sql.shuffle.partitions`,
`KAFKA-1234`-style cross-references) that a general-purpose sentence embedding compresses away
but exact lexical overlap (BM25) rewards directly. JRASERVER's larger gap likely compounds with
its already-noted sparse-label problem (`docs/phase0_framing.md`) — a general-domain product
suggestion tracker has less of the identifier-dense vocabulary that would let a text-similarity
signal generalize past raw term overlap either.

**This is a legitimate, reportable finding, not a failed phase.** Per HANDOFF: "report either
way."

**What *did* improve: latency.** Embedding retrieval is 15-50x faster end to end than BM25 at
inference time (e.g. SPARK: 40s vs 1862s for 4,680 test issues; JRASERVER: 22.5s vs 436s) —
BM25's pure-Python exhaustive scoring is exactly the cost the two-stage architecture exists to
avoid, and a FAISS index is the fix.

> **Correction (2026-07-26).** That latency gap was a property of the `rank_bm25` implementation,
> not of lexical retrieval. Precomputing BM25's document-side weights into a sparse
> term-document matrix (`src/issue_assignee_recommender/lexical.py`) gives the *same* scores at
> 0.3ms/query vs `BM25Okapi`'s 41.5ms on KAFKA — 140x, which erases the practical latency
> argument for preferring embeddings. The Recall@50 numbers in the table above are unaffected
> (same scoring function); only the "embeddings win on speed" conclusion is withdrawn.

**Both signals are worth keeping, and the Recall@50 tie was hiding that.** Phase 3 uses BM25 and
embeddings as separate candidate sources and separate ranking features, and the union retrieves
the true assignee meaningfully more often than either alone (see `docs/phase3_ranking.md`) — the
two methods miss on *different* issues even though they hit at similar rates. `bm25_affinity_score`
and `text_affinity_score` also land as the two strongest ranking features in every project's
model, neither dominating the other. Reading the Recall@50 parity above as "the two are
interchangeable, pick one" was the wrong inference; they are complementary at similar strength.

## Phase 4 gate decision: **not proceeding**

Per HANDOFF, Phase 4 (contrastive fine-tuning) is gated on Phase 2 showing headroom over the
off-the-shelf encoder. It didn't: embeddings are at parity with BM25 at best, worse at worst.
Fine-tuning an encoder to close a gap to BM25 that mostly doesn't exist would be optimizing the
wrong stage -- the HANDOFF calls this out directly ("if off-the-shelf embeddings are
near-ceiling, *not* fine-tuning and being able to say why is the better engineering call").
Logged in `docs/decisions.md`.
