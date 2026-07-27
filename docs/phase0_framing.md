# Phase 0 — Data and Task Framing

## Prediction task

At the moment a Jira issue is filed, rank the developers most likely to be the one who
eventually resolves it (i.e. becomes the `assignee` on resolution / the actor who transitions
the issue to a closed/resolved status).

## Fields legitimately available at prediction time

Available (known at or before issue creation):
- `summary`, `description` (text as originally filed — not edited versions)
- `project`, `components`, `issuetype`, `priority`, `labels` (as set at creation)
- `reporter`
- `created` timestamp
- `issuelinks` that exist as of creation (e.g. explicit "duplicates"/"relates to" set at filing)

**Leakage — excluded by construction:**
- `assignee` field itself beyond the label we're predicting
- `resolution`, `resolutiondate`, `status` transitions after creation
- comments (they're posted after creation, often by the resolver)
- changelog entries with timestamps after `created`
- any edits to `summary`/`description` made after the original creation event
- fix versions set after triage

Any feature pipeline must filter changelog/comment history to `timestamp < issue.created`
when aggregating a developer's prior activity for a given issue — this is the most likely
place to accidentally leak (see HANDOFF leakage discipline section).

## Temporal split

Train/test split is a single global date cutoff **T** per selected project, chosen so that:
- test set has enough issues for stable Recall@k/NDCG estimates (target: >=2,000 test issues)
- test set developers have enough presence in the train period to not be pure cold-start by
  construction (some cold-start is fine and expected — that's the point of Phase 5)

Train: issues with `created < T`. Test: issues with `created >= T`, evaluated strictly using
only pre-T information for feature computation (a test issue's own text is fine to use; a
developer's workload/recency features for scoring a test issue must only look at train-period
and pre-issue activity).

## Project selection criteria (to run once data is loaded)

For each of the 16 instances/projects, compute:
1. Issue count and date range
2. Number of distinct resolvers with >= N resolved issues
3. Gini coefficient / concentration of resolution across developers — reject projects where
   top-2 developers resolve >50% of issues (popularity baseline becomes unbeatable and the
   whole exercise stops being informative)
4. Fraction of issues with a non-empty, resolved assignee (need enough labeled ground truth)

Select 3-5 projects that pass these filters, prioritizing ones from the Apache instance given
domain relevance to Jira/Atlassian's own product.

## Deliverable checklist

- [x] Dataset loads, collection/field names confirmed against actual dump (not assumed)
- [x] Per-project stats table (issue count, developer count, resolver concentration, date range)
- [x] 5 projects selected with justification (see `docs/decisions.md`)
- [x] Frozen train/test split (date T per project) written to `data/processed/`
- [x] This doc updated with actual numbers once computed

## Actual results

Dataset: [The Public Jira Dataset](https://zenodo.org/records/15719919) (Montgomery et al.,
MSR 2022), restored from the MongoDB archive into a local `mongod`. Each of the 16 top-level
collections is one Jira *instance*; `fields.project.key` is the actual sub-project. Full Jira
REST API v2 issue objects, changelog histories, and comments are nested per document. Authors
(`assignee`/`reporter`/`creator`/changelog/comment authors) are anonymized via a stable UUID4
in the `.key` sub-field — consistent across documents, safe to use as developer identity.

Selected projects (see `docs/decisions.md` for the full selection rationale). Split is a
**per-project** 80th-percentile-of-`created` cutoff, not one global date — see
`data/processed/split_manifest.csv`:

| project | instance | issues | train | test | test w/ label | distinct resolvers (train/test) |
|---|---|---|---|---|---|---|
| JRASERVER | Jira | 47,225 | 37,780 | 9,445 | 856 | 399 / 198 |
| SPARK | Apache | 37,154 (289 dropped, bad timestamp) | 29,723 | 7,431 | 4,680 | 1,516 / 429 |
| HADOOP | Apache | 15,797 | 12,637 | 3,160 | 1,924 | 847 / 389 |
| KAFKA | Apache | 12,312 | 9,849 | 2,463 | 1,056 | 580 / 246 |
| CASSANDRA | Apache | 17,115 | 13,692 | 3,423 | 1,921 | 592 / 257 |

**Known data-quality note:** JRASERVER (Jira's own tracker) has far sparser ground truth than
the Apache projects — only ~25% of its issues ever get both an assignee and a resolution date
(it's partly a public suggestions/feature-request tracker; sampled docs show issues sitting in
states like "Gathering Interest" with `assignee=null` indefinitely). Its test set therefore has
only 856 labeled evaluation issues despite 9,445 total test issues. Still worth keeping for the
Atlassian-domain narrative, but its metrics will be noisier than the four Apache projects —
report per-project numbers, don't average them away.

Exported to `data/processed/{issues,changelog,comments}.parquet`, with `issues_split.parquet`
adding the frozen `split` (train/test) column. Comment bodies were dropped on export (author +
timestamp kept) — not needed for retrieval/ranking features and the largest contributor to
dump size.
