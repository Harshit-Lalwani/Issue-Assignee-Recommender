# Walkthrough: Issue-Assignee Recommender

This explains the problem, the data, every technical decision, and the actual measured
results — written for someone who has never touched Jira, recommender systems, or any of the
ML tooling used here. Nothing below is simplified to the point of being wrong; where a concept
matters, it's explained properly rather than glossed over.

---

## 1. The problem, in plain terms

Jira is issue-tracking software — companies use it to record bugs, feature requests, and tasks
("issues"), and to track who is working on what. When someone files a new issue, somebody has
to decide who should fix it. In a small team you just ask around. In a project with hundreds of
contributors and years of history (the projects here range from 12,000 to 47,000 issues each),
that decision is a real cost: someone has to read the issue, guess who has the right expertise
or is free to take it, and assign it — or the issue sits untriaged.

This project builds a system that does that automatically: **given the text of a newly-filed
issue, produce a ranked list of the developers most likely to be the one who actually resolves
it.** That's it. Not "who should work on it" in some idealized sense — literally, "based on
everyone who has resolved issues on this project before, who does this new issue most resemble
the work of."

This is useful because Jira, the real product, already ships an "assignee suggestion" feature.
This project rebuilds a version of that from scratch, on real historical data, and measures how
well it actually works — including being honest when a technique *doesn't* help.

## 2. Why this isn't trivial

Two things make this harder than it sounds:

**It's a moving target.** The set of active developers changes over time — people join
projects, go inactive, switch teams. A model has to work from *only what was knowable at the
time the issue was filed*, or the evaluation is meaningless (you'd be letting the model see the
future). This is called **leakage**, and avoiding it is the single most important discipline in
this project — explained in detail in section 5.

**The obvious answer is often wrong.** The naive approach — "just assign it to whoever resolves
the most issues" — sounds reasonable but usually fails badly, because different issues need
different expertise. Proving *quantitatively* that this naive approach is beatable, before
building anything fancier, is exactly what Phase 1 (section 6.2) does.

## 3. Where the data comes from

The data is **The Public Jira Dataset** (Montgomery, Lüders, Maalej — published at MSR 2022, a
real academic Mining Software Repositories conference), a research dataset built by
downloading real issues from 16 public Jira installations via their public APIs. It includes
Apache's Jira (the one used by dozens of major open-source projects like Spark, Kafka, Hadoop),
and — notably — **Jira's own bug tracker**, `jira.atlassian.com`, where Atlassian's own
engineers track bugs in Jira itself.

The full dataset is a 5.8GB compressed MongoDB database dump containing **2.7 million issues**
across all 16 installations. Each issue is stored as the *exact JSON object Jira's own REST API
would return for it* — the same `summary`, `description`, `assignee`, `reporter`, `status`,
`components`, full change history (`changelog`), and comments a real Jira server exposes. This
project pulled that dump down, restored it into a real local MongoDB server, and inspected the
actual schema by hand rather than guessing at it from documentation — the dataset's field
names and nesting turned out to differ from what the paper's abstract implied, which is a
useful reminder that "read the actual data" beats "read about the data" every time.

**Privacy:** every person's identity (`assignee`, `reporter`, everyone in the change history and
comments) is replaced with a random-looking UUID token like
`<<|author_key|41ab233b-e205-4bf6-b1ca-05f9dd706417|>>`. Critically, the *same* real person
always gets the *same* UUID everywhere they appear — so you can't tell who anyone actually is,
but you can still tell "this is the same developer who resolved these other 40 issues," which
is exactly the signal this project needs and the only thing it needs.

### The 5 projects used

The full dataset covers 1,822 sub-projects across those 16 installations — far too many to use
all of. Using all of them would also be scientifically pointless for some of them: if two
people resolve 90% of a project's issues, then "guess one of those two" wins by default and
no amount of modeling teaches you anything. So the first real engineering step was a data
analysis: for every project with at least 2,000 issues and 20+ distinct resolvers, compute what
fraction of resolved issues its *top 2 most active developers* account for. Reject anything
where that's over 35% (the danger zone starts around 50%).

Five projects were chosen from what passed that filter:

| Project | What it is | Issues | Distinct resolvers | Top-2 devs' share of resolutions |
|---|---|---:|---:|---:|
| **JRASERVER** | Jira's own bug tracker (Atlassian's product, on itself) | 47,225 | 484 | 12.3% |
| **SPARK** | Apache Spark (distributed data processing) | 37,154 | 1,783 | 6.8% |
| **HADOOP** | Apache Hadoop (distributed storage/compute) | 15,797 | 995 | 8.2% |
| **KAFKA** | Apache Kafka (distributed message streaming) | 12,312 | 601 | 10.1% |
| **CASSANDRA** | Apache Cassandra (distributed database) | 17,115 | 698 | 16.2% |

That's 129,603 real issues, from real, large, extremely well-known open-source projects,
each with contribution spread across hundreds to nearly two thousand different people — a
genuinely hard, genuinely realistic setting, not a toy.

## 4. The architecture: why two stages?

The system is built as **five separate models**, one per project — not one model trying to
handle all five. A developer active on Kafka has nothing to do with Cassandra; pooling them
would just add noise, and running five independent, comparable experiments is closer to how a
real company would actually deploy this (a per-team or per-repo model, not one giant global
model).

For each project, ranking a new issue happens in **two stages**:

**Stage 1 — Retrieval.** Out of hundreds or thousands of developers who have ever touched this
project, quickly narrow down to a shortlist of ~110-170 plausible candidates, using text
similarity to past issues and a few cheap activity heuristics. This has to be cheap, because it
runs against the *entire* history of the project every time.

**Stage 2 — Ranking.** Take that shortlist and score each candidate using a richer,
more expensive model that looks at behavioral signals: how much work has this person done on
this project, in this specific area, how recently, how busy are they right now, etc. This step
is expensive per-candidate, which is exactly why stage 1 exists — you can't afford to run it
against every developer who ever existed on the project, only the ~110-170 who made it through
retrieval.

This retrieval-then-rank pattern is the standard architecture for real-world recommender
systems (it's how Netflix, Spotify, and Jira's real assignee-suggestion feature all work) —
splitting a "search the whole universe" problem from a "carefully judge these few" problem,
because you cannot afford to run the expensive judgment model against everything.

**The catch that dominates this entire project, and that section 6.4 is mostly about:** stage 1
is a *filter*, and anything it throws away is gone permanently. If the person who actually
resolved the issue doesn't make the shortlist, stage 2 cannot rank them first, cannot rank them
at all, and no amount of improving stage 2 will ever recover them. That makes the system's
accuracy a product of two separate things — how often the right person makes the shortlist, and
how well the ranker orders the shortlist — and the two failure modes need completely different
fixes. Measuring them separately turned out to be the single highest-value thing done in this
project.

## 5. The leakage discipline

This is worth its own section because it's the thing most likely to silently invalidate a
result like this if done carelessly.

**The rule:** a model scoring a new issue may only use information that existed *before that
issue was filed*. Nothing about who eventually resolved it, what they said in comments, what
the resolution status later became, or anything from the issue's own future is allowed as an
input — only as the answer key used afterward to check if the model was right.

This is enforced with a **temporal split**, done independently for each project: sort all of a
project's issues by creation date, and pick a cutoff date `T` such that 80% of issues came
before it and 20% after. Everything before `T` is "train" (used to build the model);
everything at or after `T` is "test" (used only to check the model's answers, never to train
it). Because each project has a different history length (JRASERVER spans 2002-2022, KAFKA only
2011-2022), the cutoff date is different for each project — computed from that project's own
data, not one global date picked by hand.

| Project | Split date | Train issues | Test issues |
|---|---|---:|---:|
| JRASERVER | 2016-07-13 | 37,780 | 9,445 |
| SPARK | 2019-12-29 | 29,723 | 7,431 |
| HADOOP | 2017-08-18 | 12,637 | 3,160 |
| KAFKA | 2020-04-14 | 9,849 | 2,463 |
| CASSANDRA | 2017-08-22 | 13,692 | 3,423 |

It goes one level deeper in Phase 3 (section 6.4), where the model needs *labeled training
examples*, not just a fixed snapshot to compute stats from. A second, earlier cutoff inside the
training period separates "history used to compute a developer's stats" from "issues the model
is actually trained to predict," so that even a *training* example's features can't see
anything that happened at or after that example's own creation.

**Warning signs that would mean this discipline had failed**, and that were explicitly checked
for at every phase: a model doing suspiciously *too* well (near-100% accuracy), test performance
beating training performance, or one single feature dominating a model's decisions entirely.
None of those appeared anywhere in this project's results — a genuinely good sign, not just an
assumption.

A fourth check was added later, and it's the sharpest of the four: when a change produces a
large jump, verify *which* of the two stages the jump came from (section 6.4.4). A leak in the
ranking features would show up as the ranker suddenly converting more of its shortlist into
correct answers. A genuine retrieval improvement shows up as a bigger shortlist with the
conversion rate unchanged. Those look identical in the headline metric and completely different
one level down.

## 6. Walking through what was actually built and measured

### 6.1 — Getting the data usable

The raw MongoDB documents are deeply nested JSON (issue → changelog → list of history entries →
list of field changes; issue → list of comments). Querying that repeatedly for machine learning
would be slow and awkward, so the 5 projects' issues, changelogs, and comments were flattened
and exported to **Parquet** files — a columnar binary file format built for fast analytical
reads (much faster to load into a dataframe than replaying JSON parsing every time), the
standard storage format for this kind of data work. **DuckDB** — an embedded SQL engine that
reads Parquet directly with no server to set up — was used to compute the train/test split
cutoffs per project.

### 6.2 — Phase 1: proving the "obvious" answer is beatable

Before building anything sophisticated, three simple baselines were implemented and measured,
because if none of them beat "assign it to whoever's busiest," nothing built afterward would
mean anything either:

1. **Popularity** — rank every developer by how many issues they've resolved on this project,
   full stop. Same ranking for every single issue, regardless of what it's about.
2. **Component-owner** — Jira issues can be tagged with a "component" (e.g. "network",
   "storage"). Rank developers by how often they've resolved issues in *this issue's*
   component(s).
3. **BM25 k-nearest-neighbors** — BM25 is a decades-old, still extremely strong algorithm for
   scoring how relevant a document is to a search query, based on matching words (with
   adjustments for word rarity and document length) — it's the same family of algorithm that
   traditional search engines were built on before neural methods. Here, the "query" is the new
   issue's text, and the "documents" are all past issues; find the ~50 most textually similar
   past issues, and recommend whoever resolved those, weighted by how similar each one was.

Measured with four standard recommender-system metrics: **Recall@5** (is the true resolver
somewhere in the top 5 guesses?), **Recall@10**, **MRR** (Mean Reciprocal Rank — rewards
getting it right in position 1 much more than position 10), and **NDCG@10** (Normalized
Discounted Cumulative Gain — a graded version of the same idea, standard in search/ranking
research).

**Result: BM25 beat both other baselines on every single project, on every single metric.**

| Project | Popularity NDCG@10 | Component-owner NDCG@10 | BM25 NDCG@10 |
|---|---:|---:|---:|
| JRASERVER | 0.022 | 0.091 | **0.115** |
| SPARK | 0.070 | 0.112 | **0.213** |
| HADOOP | 0.089 | 0.170 | **0.175** |
| KAFKA | 0.119 | 0.184 | **0.204** |
| CASSANDRA | 0.063 | 0.084 | **0.111** |

This is the single most important checkpoint in the whole project: it proves the "obvious"
answer is genuinely beatable here, which justifies everything built after it. (Note also how
weak the popularity baseline is on JRASERVER specifically — Recall@5 of 0.7%. That project spans
20 years, and the people most active in 2002-2016 mostly aren't the people active in 2016-2022;
a flaw hiding in the data itself, honestly reported rather than smoothed over.)

### 6.3 — Phase 2: does understanding meaning (not just words) help?

BM25 matches literal words. **Sentence embeddings** are a different approach: a neural network
(here, `all-MiniLM-L6-v2`, a small, fast, freely available model — this is what "off-the-shelf"
means, as opposed to a model trained specifically for this data) reads a piece of text and
outputs a vector of 384 numbers that capture its *meaning* — texts about similar topics end up
with similar vectors, even if they don't share many exact words. Searching for the closest
vectors to a new issue's vector, out of tens of thousands of past issues' vectors, needs to be
fast — that's what **FAISS** (Facebook AI Similarity Search) provides: an exact, brute-force
nearest-neighbor search library. ("Exact" and "flat" here specifically means it checks every
vector rather than an approximate shortcut — at this scale (5,000-18,000 vectors per project)
that's fast enough that the approximate versions (IVF/HNSW) would only add complexity for zero
speed benefit, a real engineering call, not a default.)

**Metric: Recall@50** — is the true resolver anywhere in a shortlist of the top 50 candidates?
This is the metric that matters for stage 1 specifically: it doesn't need to be *ranked*
correctly yet, it just needs to *not be thrown away* before stage 2 gets a chance to rank it.

| Project | BM25 Recall@50 | Embeddings Recall@50 |
|---|---:|---:|
| JRASERVER | **0.317** | 0.225 |
| SPARK | 0.511 | **0.513** |
| HADOOP | **0.338** | 0.323 |
| KAFKA | 0.482 | **0.491** |
| CASSANDRA | **0.386** | 0.372 |

**Embeddings did not clearly beat BM25.** They're statistically tied on 2 projects and
meaningfully worse on JRASERVER. The likely reason: these issue trackers are full of exact
identifiers — stack traces, class names, config keys like `spark.sql.shuffle.partitions`,
cross-references like `KAFKA-1234` — which BM25's literal word-matching rewards directly, but a
general-purpose meaning-embedding tends to smooth over. This is a real, useful, negative-ish
result, not a failure: it's reported plainly (see `docs/phase2_retrieval.md`) rather than
buried, and it directly justifies a decision — **not** doing Phase 4 (fine-tuning the embedding
model on this project's own data), because fine-tuning would be spending real GPU time trying
to close a gap to BM25 that mostly doesn't exist. Recognizing when *not* to do a fancier
technique is as much a real engineering decision as building one.

**Two conclusions originally drawn from this table were later found to be wrong, and were
retracted rather than quietly left in place.** Both are worth reading as examples of how a
plausible-sounding measurement can mislead:

*Retraction 1 — "embeddings are 15-50x faster than BM25."* That was measured, and the numbers
were real (Spark: 40 seconds vs. 31 minutes for the same test set). But it was comparing FAISS,
a purpose-built index, against `rank_bm25` — a pure-Python library that rescores *every document
in the corpus, one at a time, in a Python loop* for every single query. That's a property of
that library, not of BM25 the algorithm. Rewriting BM25 so its document-side weights are
computed once up front and stored in a **sparse matrix** (a matrix that stores only its non-zero
entries — the right structure here because any given issue uses only a few hundred of the
corpus's tens of thousands of distinct words, so the matrix is >99% zeros) turns each query into
a single matrix-vector multiplication handled by optimized C code. Same algorithm, same scores,
**0.3 milliseconds per query instead of 41.5** — a 140x speedup that erases the entire latency
argument. The lesson is a general one: a benchmark comparing two *implementations* says nothing
about the two *methods* until both are implemented comparably.

*Retraction 2 — "they score about the same, so they're interchangeable; just pick one."* Two
methods hitting at the same *rate* does not mean they hit on the same *issues*. Section 6.4
shows that using both together retrieves the right person substantially more often than either
alone — they succeed and fail on different issues, so their errors partly cancel. The tie in
this table was hiding a genuine complementarity, and reading "tied" as "redundant" cost this
project real accuracy until it was caught.

### 6.4 — Phase 3: the model that actually learns, and the headline result

Both baselines and embeddings only look at *one type of signal at a time* (word overlap,
component match, or raw popularity). A real decision about who should get an issue depends on
*combining several signals* — and figuring out how much to trust each one isn't something to
guess by hand. That's what a **learned ranking model** does.

**LightGBM** is a fast, widely-used implementation of *gradient-boosted decision trees* — an
algorithm that builds many small decision trees in sequence, where each new tree specifically
targets the mistakes the trees before it made. Its `LambdaRank` mode is built specifically for
ranking problems (as opposed to plain classification or regression) — it's trained to get the
*order* of a list right, not just to score any one item correctly in isolation.

For each candidate developer on each issue, a set of numbers ("features") is computed, all
respecting the leakage cutoff from section 5:

- **`dev_project_total`** — how many issues they've resolved on this project, ever (as of the
  cutoff)
- **`dev_component_count`** — how many of those were in this issue's specific component(s)
- **`dev_reporter_affinity`** — how many times they've resolved issues filed by *this specific
  reporter* before (people who work together tend to keep working together)
- **`dev_recency_days`** — how long it's been since their last resolved issue (very inactive
  people are less likely to pick up a new one)
- **`dev_open_workload`** — how many issues are currently assigned to them and still unresolved
  (busy people get fewer new issues in a fair system)
- **`dev_recent_volume_365d`** — resolutions in the trailing 365 days, as opposed to
  `dev_project_total`'s all-time count. Added specifically to test whether decaying old activity
  would help JRASERVER's 20-year-turnover problem (see the negative-result callout below).
- **`text_affinity_score`** — the embedding-similarity score from Phase 2, carried forward as
  one input among several rather than the only signal

**The first attempt at this model produced these results — NDCG@10 vs. the best Phase 1
baseline:**

| Project | Best baseline (BM25) | LightGBM ranker (first version) | Relative improvement |
|---|---:|---:|---:|
| JRASERVER | 0.115 | 0.126 | +10% |
| SPARK | 0.213 | 0.464 | +118% |
| HADOOP | 0.175 | 0.290 | +66% |
| KAFKA | 0.204 | 0.404 | +98% |
| CASSANDRA | 0.111 | 0.229 | +106% |

Roughly doubling ranking quality on 4 of 5 projects. That looked like a good place to stop — and
that judgment turned out to be wrong, for a reason nothing in the table above could reveal.

#### 6.4.1 — Finding the actual bottleneck (the most important step in the project)

Two separate attempts to push these numbers higher had already come back with nothing. Adding a
"recent activity" feature did nothing (below). Adding an LLM re-ranker on top made results
*worse* (section 6.5). Two independent, reasonable ideas failing in a row is itself information:
it suggests the thing being improved isn't the thing that's broken.

So instead of trying a third idea, the next step was a measurement. Recall from section 4 that
accuracy here is really two numbers multiplied together:

> **final accuracy = (how often the right person makes the shortlist) x (how well the ranker
> orders the shortlist once they're on it)**

The first factor has a name — **oracle recall**, meaning "the score a hypothetical perfect
ranker would get, given this shortlist." It's an upper bound: reordering a list that doesn't
contain the answer cannot produce the answer. Neither factor had ever been measured separately;
only their product had. Measuring them takes about 4 minutes of compute, and this is what came
back:

| Project | Oracle recall (right person made the shortlist) | Ranker's actual Recall@10 | Share of what was possible |
|---|---:|---:|---:|
| JRASERVER | 0.268 | 0.244 | **91%** |
| SPARK | 0.512 | 0.493 | **96%** |
| HADOOP | 0.350 | 0.328 | **94%** |
| KAFKA | 0.507 | 0.476 | **94%** |
| CASSANDRA | 0.349 | 0.289 | **83%** |

Read the last column carefully, because it reframes everything. **The ranking model was already
finding 83-96% of the people it was given a chance to find.** Its remaining headroom was a few
percentage points. Meanwhile the shortlist was throwing away the correct person *before the
ranker ever saw them* on half to three-quarters of all issues.

Every improvement attempt so far had been aimed at the stage that was already working almost
perfectly. This is why the recent-activity feature did nothing and why the LLM re-ranker
couldn't help: both were competing for a few points of headroom in stage 2, while the actual
losses were happening in stage 1. **The model wasn't bad at ranking. It was being handed the
wrong shortlist.**

#### 6.4.2 — Fixing the shortlist

Once the problem is stated correctly, the fix is not sophisticated — which is the point. Two
things changed.

**Look further down each existing list.** The original shortlist took the resolvers of the top
50 most similar past issues, the top 20 owners of the issue's component, and the top 20 most
prolific developers overall. Those cutoffs were round numbers chosen by hand, never tested.
Widening them to 200 / 50 / 50 costs almost nothing (the ranker's cost grows linearly with
candidates, and it was never the expensive part) and recovers a large number of correct people
who were sitting just past an arbitrary line.

**Add two more ways in.** A person is now also shortlisted if they resolved any of the top 100
issues that **BM25** finds textually similar (this is where retraction 2 from section 6.3 pays
off — BM25 finds people the embeddings miss, precisely because it matches exact identifiers and
error strings rather than general meaning), or if they are among the **50 most active developers
in the trailing year**, regardless of what the issue says. That second one deserves a note: it's
a crude, almost dumb heuristic, and it was the single biggest contributor of the four changes.
Text similarity is good at "who knows about this subject" and completely blind to "who is even
around right now" — and on projects spanning a decade or more, being currently active is a huge
part of the answer.

Using BM25 as a shortlist source only became affordable *because* of the sparse-matrix rewrite
described in section 6.3 — at `rank_bm25`'s 41.5ms per query it would have been far too slow to
run on every issue; at 0.3ms it's free. A performance fix and an accuracy fix turned out to be
the same piece of work.

Here's what each change bought, measured cumulatively (oracle recall — how often the right
person is on the shortlist at all):

| Project | Original | + wider cutoffs | + BM25 | + recently-active | Realistic ceiling |
|---|---:|---:|---:|---:|---:|
| JRASERVER | 0.268 | 0.439 | 0.496 | **0.597** | 0.723 |
| SPARK | 0.512 | 0.653 | 0.671 | **0.713** | 0.815 |
| HADOOP | 0.350 | 0.441 | 0.466 | **0.506** | 0.651 |
| KAFKA | 0.507 | 0.652 | 0.663 | **0.743** | 0.815 |
| CASSANDRA | 0.349 | 0.455 | 0.483 | **0.575** | 0.738 |

That last column is important for honesty about what "perfect" would even mean here. **Between
18% and 35% of test issues are resolved by somebody who had never resolved a single issue on
that project before the cutoff date** — a brand-new contributor. No system built on historical
activity can recommend a person it has never seen do anything. That slice is unreachable by
construction, so the realistic ceiling isn't 1.0, it's `1 - (that fraction)`. The final
shortlist captures 78-91% of what is actually attainable.

**One idea was tested and rejected on the numbers:** also shortlisting people who *commented* on
similar past issues (commenting shows interest and knowledge even without a resolution). It
added only 0.3-0.7 percentage points of oracle recall while making the shortlist ~40% larger —
not worth it. The related hope that comment history could rescue the cold-start slice failed
too: only 6-13% of those never-seen-before resolvers had even commented on anything beforehand.
Both were measured before being believed, and both are documented rather than dropped.

#### 6.4.3 — Better features, too (and how much they actually mattered)

Seven features were added alongside the shortlist work — worth describing because the *kind* of
feature added is the interesting part:

- **`bm25_affinity_score`** — BM25 similarity as a ranking signal, mirroring the existing
  embedding one, so the ranker can weigh both kinds of text match.
- **`text_affinity_rank`, `bm25_affinity_rank`** — the candidate's *rank* by each similarity
  score (1st, 2nd, 3rd...) rather than the raw score. This matters more than it sounds: a raw
  similarity score isn't comparable between issues, because an issue with a long description
  matches more past issues and gets bigger scores across the board. A decision tree splitting on
  "score > 0.4" therefore means different things on different issues. Rank is scale-free — "the
  single best text match for this issue" means the same thing everywhere.
- **`dev_component_share`, `dev_reporter_affinity_share`** — the same idea applied to counts.
  "Resolved 30 issues in this component" means something very different in a component with
  3,000 historical issues than in one with 40. Dividing by the component's total converts a raw
  count into a share, which is comparable across issues.
- **`dev_component_recent_365d`, `dev_recent_volume_90d`, `dev_issuetype_count`** — component
  expertise restricted to the last year, a shorter 90-day activity window, and how often this
  person resolves this *type* of issue (Bug vs. Improvement vs. Task).

To find out how much these were actually worth, the whole pipeline was re-run a second time with
the new features but the **old, narrow shortlist**, isolating one change from the other. This is
an **ablation** — deliberately removing one part of a change to attribute credit, rather than
shipping two improvements at once and assuming both helped:

| Project | Best baseline | First version | New features, **old** shortlist | New features + new shortlist |
|---|---:|---:|---:|---:|
| JRASERVER | 0.115 | 0.126 | 0.128 | **0.153** |
| SPARK | 0.213 | 0.464 | 0.471 | **0.633** |
| HADOOP | 0.175 | 0.290 | 0.302 | **0.389** |
| KAFKA | 0.204 | 0.404 | 0.415 | **0.569** |
| CASSANDRA | 0.111 | 0.229 | 0.251 | **0.370** |

**Seven features' worth of careful engineering bought 1-2 points. The shortlist fix bought the
other 85-90% of the gain.** Reporting that split matters: without the ablation, the honest
conclusion ("we found and fixed a first-stage recall bug") would be indistinguishable from the
flattering one ("our feature engineering doubled performance"), and only one of those is true.

#### 6.4.4 — The headline result

| Project | Best baseline (BM25) | Final LightGBM ranker | Relative improvement | Recall@5 |
|---|---:|---:|---:|---:|
| JRASERVER | 0.115 | **0.153** | +34% | 0.173 |
| SPARK | 0.213 | **0.633** | +198% | 0.644 |
| HADOOP | 0.175 | **0.389** | +122% | 0.401 |
| KAFKA | 0.204 | **0.569** | +180% | 0.595 |
| CASSANDRA | 0.111 | **0.370** | +234% | 0.398 |

An unexpected bonus: despite roughly tripling the number of candidates scored per issue, the
system got *faster*. Investigating the pipeline for this work surfaced that every issue was
being fed through the neural encoder and the similarity search **twice** — once to build the
shortlist, once to compute the similarity feature — because those were two separate functions
that each independently did the expensive part. Merging them into a single pass more than paid
for the extra candidates.

**Is this too good to be true?** That question deserves a real answer rather than a reassurance,
since section 5 warns that a suspiciously large jump is exactly what leakage looks like. The
sharpest check available is the *conversion rate* — the ranker's score divided by its oracle
recall, i.e. how much of its shortlist it successfully converts. If the new features had started
secretly seeing the future, conversion would spike. It didn't move: SPARK converted 91% of its
ceiling before and 89% after; KAFKA 79% before, 77% after. **The ranker got no better at its
job — it was simply given a better shortlist to work from**, which is precisely the claim being
made and not a disguised leak. Feature importance is also spread across all 15 features with no
single dominator, and the two text-similarity features (embedding and BM25) land at comparable
strength — the direct evidence for retraction 2 in section 6.3.

**JRASERVER changed category entirely.** Its shortlist quality more than doubled (0.268 →
0.603), but its score barely moved (0.126 → 0.153). It now converts just 25% of its available
ceiling, against 64-89% everywhere else — the exact inverse of every other project. The
candidates are there now; the behavioral features simply cannot tell them apart. That's a far
more actionable diagnosis than the previous one ("sparse labels, not much to be done"), and it
supersedes it: JRASERVER's problem is a ranking problem specific to a 20-year tracker where who
was productive in 2008 says little about who is available in 2020. The next thing to try there
is time-*decayed* features that replace the all-time snapshot, rather than adding another window
feature next to it — logged in `docs/decisions.md` as the next step rather than attempted here.

**A negative result, worth including precisely because it's honest:** adding
`dev_recent_volume_365d` was a direct, reasonable-sounding fix for JRASERVER's sparse-turnover
problem — someone active a decade ago shouldn't look identical to someone steadily active now.
It was implemented and measured, not just proposed. Result: no meaningful change anywhere,
including JRASERVER (0.126 -> 0.126). With hindsight from section 6.4.1, the reason is clear: it
was a stage-2 feature aimed at a stage-1 problem. The idea itself wasn't even wrong — "who is
active recently" turned out to be highly valuable — but it only paid off once applied where the
losses actually were, as a *shortlist source* rather than as a ranking feature. Same intuition,
wrong stage, and the difference between the two was worth ~40% of the final gain.

### How good are these numbers, really — and would this be usable in production?

Context matters more than the raw metric. With 399-1,516 distinct resolvers in each project's
training history, random guessing gets a Recall@5 of roughly 0.3%-1.3%. The final numbers above
are **14x to 195x better than random** depending on the project (SPARK: ~195x; JRASERVER:
~14x). Compared against
published bug-triage research (DeepTriage and similar deep-learning models report 52%-87%
top-5 accuracy on Eclipse/Mozilla/NetBeans), this project's numbers look lower in absolute
terms — but that comparison isn't apples-to-apples: a lot of that literature restricts the
candidate pool to a small set of historically frequent fixers (sometimes as few as 20-50
people), which inflates the apparent accuracy relative to picking from a realistic full
developer pool of hundreds to nearly two thousand people, which is what this project
deliberately kept (the whole point of the project-selection filter in section 3).

Whether it's "usable" depends on the deployment mode. **As an assistive suggestion** — narrowing
a human triager's search from hundreds of people down to 5, correct 40%-64% of the time on 4 of
5 projects — yes, genuinely useful, the same way GitHub/Gerrit reviewer-suggestion features are
used (a human still confirms the final pick). SPARK and KAFKA, at ~60% top-5 accuracy against a
pool of hundreds, are in the range where a triager would plausibly click a suggestion more often
than not. **As a fully autonomous auto-assigner with no human check** — still no: even at the
new numbers the single top guess is wrong most of the time, and JRASERVER (17% top-5) isn't good
enough for unsupervised use in any mode.

There's also a principled way to know *when* the system should decline to answer, which falls
out of section 6.4.1's measurement. Between 18% and 35% of issues are resolved by someone with
no prior history on the project — for those, the correct output is not a confident top-5 but an
admission that the model has never seen the right person. That slice is now quantified per
project rather than hidden inside an averaged error rate, which is what would make an
"abstain instead of guessing" behavior implementable rather than aspirational.

**What does Atlassian actually ship?** Worth checking rather than assuming. Per Atlassian's own
support documentation, classic Jira's assignee-suggestion feature is simpler than what's built
here: it just shows the last 5 people *the current reporter* has personally assigned issues to
before, filtered to permissions — a personalized-recency heuristic with no text analysis,
closest in spirit to this project's `dev_reporter_affinity` feature alone. Jira Service
Management has a separate, newer AI-powered suggestion feature that does read request content,
closer to this project's approach, but Atlassian doesn't publish its internals.

### 6.5 — Phase 5 (concluded, negative result): can an LLM re-ranker do even better on hard cases?

Even a good ranking model struggles most on "cold-start" cases — issues whose true resolver has
very little track record on the project yet (a new contributor, or someone picking up an
unfamiliar area). The idea being tested: take the ranking model's top 10 guesses for those hard
cases, and ask a large language model to re-read the actual issue text plus each candidate's
stats and *reconsider the order*, since an LLM might catch subtler textual/contextual signals a
handful of numeric features can't.

Four LLM providers were wired up with a fallback chain (if one fails or is rate-limited, the
next is tried automatically) — **Groq** (specialized fast-inference hardware; extremely low
latency), **OpenRouter** (a marketplace that proxies many different providers' models through
one API), **Google Gemini**, and **NVIDIA NIM** (NVIDIA's hosted catalog of 100+ open models) —
with real API keys, live model catalogs checked directly against each provider's API rather
than assumed, and each candidate model actually called and measured rather than picked by
reputation.

**The honest result so far: re-ranking made things *worse*, not better.** On a real sample of
21 held-out test issues where the true resolver had sparse history (2 or fewer past
resolutions) but was still present in the ranking model's own top 10:

| Model | Valid JSON responses | Recall@5 before re-ranking (LightGBM alone) | Recall@5 after LLM re-ranking |
|---|---:|---:|---:|
| Groq `llama-3.3-70b-versatile` | 21/21 | 0.905 | **0.524** |
| Gemini `gemini-3.1-flash-lite` | 16/21 (5 hit a quota limit) | 0.905 | **0.429** |

Even after fixing the first attempt's problems (adding retry logic for rate limits, and
enriching the prompt with each candidate's actual most-relevant past issue text instead of just
summary statistics), the result held up: the already-trained ranking model's ordering was
*better* than what the LLM produced when asked to reconsider it. This makes sense in
retrospect — LightGBM was trained end-to-end, with real labeled examples, specifically to get
this exact ranking task right; an LLM doing one-shot reasoning over a text summary of the same
underlying signals doesn't have that advantage, no matter how capable the model is in general.

**A postscript from section 6.4.1 that makes this result less surprising than it first looked.**
Notice the "before" column: 0.905. The experiment, by construction, only included issues where
LightGBM had *already* put the right person in its top 10 — so the LLM was being asked to
improve on a list that was right 90% of the time, where almost the only available outcome is to
break something that already worked. That framing was invisible at the time and obvious in
hindsight: the same measurement blind spot that made the ranker look like the thing worth
improving also made this experiment look like a promising place to spend money. The LLM result
is still a real negative finding, but the more useful lesson is that it was aimed at the wrong
stage — a fifth of the effort spent measuring where the losses actually were would have
redirected it.

This is being reported as a real, negative finding rather than hidden or reframed — exactly the
kind of result the original project brief explicitly calls a legitimate outcome ("a negative
result here is explicitly acceptable and should be reported as one"). **Phase 5 is closed on
this result**: the LLM re-ranking client (`src/issue_assignee_recommender/llm.py`) stays in the
repo as the artifact of a real experiment with real API calls against four live providers, but
it is not wired into the serving API, and won't be by default — shipping a component that's
been measured to make results worse would defeat the point of measuring it in the first place.
See `docs/decisions.md` for the full writeup.

### 6.6 — Phase 6: making it a real, running service

A trained model sitting in a script isn't a product. The trained artifacts (the FAISS index,
all the aggregate stats, and the LightGBM model itself, one full set per project) are saved to
disk, and a **FastAPI** web server (FastAPI is a modern Python framework for building HTTP
APIs, chosen for being fast and having automatic request validation) loads them all once at
startup — taking about 10 seconds, instead of the 3-8 minutes it would take to retrain from
scratch on every server restart.

Sending a real, previously-unseen-style issue to the running server —

```
POST /recommend/KAFKA
{"summary": "Consumer group rebalance causes duplicate message processing", ...}
```

— returns a ranked list of anonymized developer IDs with model confidence scores, scoring ~120
candidates in about **42 milliseconds** (median; 60ms at the 95th percentile), measured on the
real saved artifacts rather than estimated.

That figure was ~500ms before the Phase 3 rework, and the improvement came from removing
duplicated work rather than from optimization: as noted in section 6.4.4, each request was
encoding the issue text through the neural network and searching the vector index twice, because
building the shortlist and computing the similarity feature were separate functions that each
did the expensive part independently. Merging them into one pass cut the dominant cost roughly
in half and more than absorbed the extra candidates. The BM25 stage that was *added* in the same
change contributes 0.3ms — negligible next to the neural encoder, and only because of the
sparse-matrix implementation from section 6.3.

## 7. The technology stack, summarized

| Tool | What it actually is | Why it's here |
|---|---|---|
| MongoDB | A document (JSON-based) database | The dataset ships as a MongoDB dump; needed to restore and query it |
| Parquet | A columnar binary file format | Fast, compact storage for the flattened, cleaned data |
| DuckDB | An embedded analytical SQL engine | Fast SQL directly over Parquet files, no server needed |
| BM25 | A classic term-matching relevance-scoring algorithm | Baseline #3, still competitive with modern methods here, and a shortlist source in Phase 3 |
| SciPy sparse matrices | Storage for matrices that are almost all zeros | Re-implementing BM25 as one matrix multiply per query — 140x faster than the pure-Python library, which is what made it affordable to run on every issue |
| Sentence-Transformers / MiniLM | A small neural network producing meaning-vectors for text | Semantic (not just literal) text similarity |
| FAISS | A vector nearest-neighbor search library | Fast lookup of "most similar past issues" at scale |
| LightGBM | A gradient-boosted decision tree library | The actual learned ranking model (Phase 3) |
| Groq / Gemini / OpenRouter / NVIDIA NIM | LLM inference providers | Candidate cold-start re-ranking backends (Phase 5) |
| FastAPI | A Python web API framework | Serves the trained model as a real HTTP endpoint |

## 8. Current status

Phases 0 (data), 1 (baselines), 2 (retrieval), 3 (learned ranking), and a working version of 6
(serving) are complete and measured on real data, with every number in this document pulled
directly from `data/processed/*.csv` and the phase documents in `docs/`. Phase 4 (fine-tuning
the embedding model) was deliberately not done, because Phase 2 showed no headroom to justify
it — and section 6.4's finding argues further against it, since the gains came from retrieving
more *broadly*, not from matching more *precisely*. Phase 5 (LLM cold-start re-ranking) is
**closed with a negative result**: benchmarked for real against four LLM providers, it made
rankings worse rather than better, and per that result it is not wired into serving.

Current per-project numbers, and the ceiling each is working against:

| Project | Recall@5 | NDCG@10 | Right person on the shortlist | Unreachable (never-seen contributor) |
|---|---:|---:|---:|---:|
| JRASERVER | 0.173 | 0.153 | 0.603 | 28% |
| SPARK | 0.644 | 0.633 | 0.710 | 19% |
| HADOOP | 0.401 | 0.389 | 0.504 | 35% |
| KAFKA | 0.595 | 0.569 | 0.744 | 18% |
| CASSANDRA | 0.398 | 0.370 | 0.576 | 26% |

**Known remaining headroom, stated concretely rather than as "future work":**

1. **Cold start is now the largest single bucket of error** — 18-35% of issues per project have
   a resolver the system has never seen resolve anything. Ranking improvements cannot touch this;
   it needs either a different signal (activity that isn't resolution) or an explicit abstention.
2. **Shortlist quality is still 12-15 points below its realistic ceiling** on every project.
3. **JRASERVER is ranking-limited, not retrieval-limited** (section 6.4.4) — the one project
   where feature work, specifically time-decayed features, is the right next move.

Three negative or retracted results are reported here as findings rather than hidden: the LLM
re-ranker that made things worse (6.5), the trailing-window feature that did nothing as a
ranking feature though the same intuition later paid off as a shortlist source (6.4.4), and two
Phase 2 conclusions that were measured again and withdrawn (6.3). The last of these is the one
worth taking most seriously, because it was wrong in a costly direction: believing BM25 was slow
and redundant delayed the fix that ultimately produced most of this project's improvement.

For the decision-by-decision reasoning behind every judgment call in this list (project
selection, split methodology, why Phase 4 was skipped, the Phase 5 conclusion, the candidate-
generation rework and both Phase 2 retractions), see `docs/decisions.md`. For the full measured
writeup of any individual phase, see the correspondingly-named file in `docs/`.
