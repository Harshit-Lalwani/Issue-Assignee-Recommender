"""Phase 5 prep: benchmark candidate LLMs for cold-start re-ranking before picking defaults.

Methodology: for each project, take test issues where the true resolver has sparse history
(dev_project_total <= 2 in the serving-time FeatureBuilder) -- a proxy for the "new
contributor" cold-start slice HANDOFF Phase 5 targets. Get the LightGBM top-10 shortlist (same
artifacts scripts/08 already trained), then ask each candidate LLM to re-rank those 10 given
the issue text + each candidate's stats. Compare: does re-ranking move the true resolver up
relative to the LightGBM-only order, and at what latency/reliability cost?

Only real API calls against real held-out test issues -- no synthetic prompts.
"""
import json
import re
import sys
import time

import lightgbm as lgb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "/root/Issue-Assignee-Recommender/src")

from issue_assignee_recommender import llm
from issue_assignee_recommender.data import PROJECTS, labeled, load_project
from issue_assignee_recommender.features import FEATURE_COLUMNS, FeatureBuilder
from issue_assignee_recommender.metrics import recall_at_k, reciprocal_rank
from issue_assignee_recommender.text import tokenize


def _s(x) -> str:
    """Parquet nulls surface as float('nan'), not None -- 'nan or \"\"' would keep the NaN."""
    if x is None or (isinstance(x, float) and x != x):
        return ""
    return str(x)


def best_matching_past_issue(query_tokens: set, dev_issues: list[tuple[str, str]]) -> str | None:
    """dev_issues: list of (summary, description) this developer resolved in history.
    Returns the summary of whichever one has the most word overlap with the query -- a cheap
    stand-in for "retrieve this candidate's most relevant past work" without a per-dev index."""
    if not dev_issues:
        return None
    best_score, best_summary = -1, None
    for summary, description in dev_issues:
        tokens = set(tokenize(f"{_s(summary)} {_s(description)}"))
        score = len(query_tokens & tokens)
        if score > best_score:
            best_score, best_summary = score, summary
    return best_summary

MODEL_DIR = "/root/Issue-Assignee-Recommender/data/models"
N_PER_PROJECT = 5

SYSTEM_PROMPT = (
    "You are helping route a newly-filed software issue to the developer most likely to "
    "resolve it. You will see the issue text and a shortlist of candidate developers "
    "(anonymized IDs) with their historical stats on this project. Read the issue text "
    "carefully -- prior candidate stats are sparse for some developers on purpose, so text "
    "similarity to the kind of work each developer's stats suggest matters. "
    "Respond with ONLY a JSON array of the candidate IDs, reordered from most to least likely "
    "to resolve this issue. Include all given IDs exactly once. No prose, no markdown fences."
)


def build_user_prompt(issue_text: str, candidates: list[dict]) -> str:
    lines = [f"ISSUE:\n{issue_text[:1500]}\n\nCANDIDATES:"]
    for c in candidates:
        example = c.get("example_issue")
        example_line = f' | most_relevant_past_issue="{example[:150]}"' if example else " | no prior resolved issues in this project"
        lines.append(
            f"- id={c['id']} | resolved_in_project={c['dev_project_total']} | "
            f"resolved_in_component={c['dev_component_count']} | "
            f"days_since_last_active={c['dev_recency_days']:.0f} | "
            f"open_workload={c['dev_open_workload']} | text_similarity={c['text_affinity_score']:.3f}"
            f"{example_line}"
        )
    lines.append("\nReturn the JSON array of ids now.")
    return "\n".join(lines)


def parse_ranked_ids(text: str, valid_ids: set) -> list:
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if not match:
        return []
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return []
    return [p for p in parsed if p in valid_ids]


def get_shortlists(project_key, model):
    train, test = load_project(project_key)
    train_l = labeled(train)
    test_l = labeled(test)

    fb = FeatureBuilder.load(f"{MODEL_DIR}/{project_key}", model)
    booster = lgb.Booster(model_file=f"{MODEL_DIR}/{project_key}_lgbm.txt")

    dev_issues: dict[str, list[tuple[str, str]]] = {}
    for row in train_l.itertuples():
        dev_issues.setdefault(row.assignee_key, []).append((row.summary, row.description))

    examples = []
    for row in test_l.itertuples():
        issue_row = {
            "summary": row.summary,
            "description": row.description,
            "components": row.components,
            "reporter_key": row.reporter_key,
            "issuetype": row.issuetype,
        }
        true_dev = row.assignee_key
        true_dev_volume = fb.popularity.get(true_dev, 0)
        if true_dev_volume > 2:
            continue  # not a sparse-history / cold-start-like example

        ctx = fb.score_issue(issue_row)
        cand = ctx["candidates"]
        if true_dev not in cand:
            continue  # retrieval never surfaced the true dev; re-ranking can't help here anyway

        devs = list(cand)
        X = [[fb.feature_row(issue_row, d, ctx)[c] for c in FEATURE_COLUMNS] for d in devs]
        scores = booster.predict(X)
        ranked = sorted(zip(devs, scores), key=lambda x: -x[1])[:10]
        if true_dev not in [d for d, _ in ranked]:
            continue  # true dev outside LightGBM's own top 10; not a re-ranking scenario

        anon_map = {f"D{i+1}": d for i, d in enumerate([d for d, _ in ranked])}
        rev_map = {d: a for a, d in anon_map.items()}
        query_tokens = set(tokenize(_s(issue_row["summary"]))) | set(tokenize(_s(issue_row["description"])))
        cand_stats = []
        for d, _ in ranked:
            feats = fb.feature_row(issue_row, d, ctx)
            example = best_matching_past_issue(query_tokens, dev_issues.get(d, []))
            cand_stats.append({"id": rev_map[d], "example_issue": example, **feats})

        examples.append(
            {
                "issue_text": f"{_s(row.summary)}\n{_s(row.description)}",
                "candidates": cand_stats,
                "lgbm_order": [rev_map[d] for d, _ in ranked],
                "true_id": rev_map[true_dev],
            }
        )
        if len(examples) >= N_PER_PROJECT:
            break
    return examples


def eval_model_on_examples(provider_fn, label, examples):
    results = []
    for ex in examples:
        prompt = build_user_prompt(ex["issue_text"], ex["candidates"])
        valid_ids = {c["id"] for c in ex["candidates"]}
        t0 = time.time()
        try:
            text = provider_fn(SYSTEM_PROMPT, prompt)
            latency = time.time() - t0
            ranked = parse_ranked_ids(text, valid_ids)
            ok = len(ranked) == len(valid_ids)
        except Exception as e:  # noqa: BLE001
            latency = time.time() - t0
            ranked, ok = [], False
            print(f"    [{label}] ERROR: {e}", file=sys.stderr)

        lgbm_rank = ex["lgbm_order"].index(ex["true_id"]) + 1
        if ok:
            llm_rank = ranked.index(ex["true_id"]) + 1 if ex["true_id"] in ranked else None
        else:
            llm_rank = None

        results.append(
            {
                "latency_s": latency,
                "json_ok": ok,
                "lgbm_rank": lgbm_rank,
                "llm_rank": llm_rank,
                "recall5_lgbm": 1 if lgbm_rank <= 5 else 0,
                "recall5_llm": 1 if (llm_rank is not None and llm_rank <= 5) else 0,
                "mrr_lgbm": 1.0 / lgbm_rank,
                "mrr_llm": (1.0 / llm_rank) if llm_rank else 0.0,
            }
        )
    return results


def summarize(label, results):
    n = len(results)
    ok_n = sum(r["json_ok"] for r in results)
    avg_latency = sum(r["latency_s"] for r in results) / n
    mrr_lgbm = sum(r["mrr_lgbm"] for r in results) / n
    mrr_llm = sum(r["mrr_llm"] for r in results) / n
    recall5_lgbm = sum(r["recall5_lgbm"] for r in results) / n
    recall5_llm = sum(r["recall5_llm"] for r in results) / n
    print(
        f"{label:35s} n={n:3d} json_ok={ok_n}/{n} avg_latency={avg_latency:6.2f}s  "
        f"MRR lgbm={mrr_lgbm:.3f}->llm={mrr_llm:.3f}  Recall@5 lgbm={recall5_lgbm:.3f}->llm={recall5_llm:.3f}"
    )


def main():
    print("loading embedding model + shortlists ...", file=sys.stderr)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")

    all_examples = []
    for project_key in PROJECTS:
        ex = get_shortlists(project_key, model)
        print(f"[{project_key}] {len(ex)} cold-start-like examples", file=sys.stderr)
        all_examples.extend(ex)

    print(f"\nTotal benchmark examples: {len(all_examples)}\n")

    candidates_to_test = [
        ("groq: llama-3.3-70b-versatile", lambda s, u: llm.call_groq(s, u, "llama-3.3-70b-versatile")),
        ("gemini: gemini-3.1-flash-lite", lambda s, u: llm.call_gemini(s, u)),
    ]
    for label, fn in candidates_to_test:
        results = eval_model_on_examples(fn, label, all_examples)
        summarize(label, results)


if __name__ == "__main__":
    main()
