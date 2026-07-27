"""Phase 3 deliverable: LightGBM LambdaRank ranking model, NDCG@10 lift over the best
Phase 1 baseline (BM25 kNN).

Leakage boundary (the highest-risk part of this project, per HANDOFF):
  - Test issues are scored using a FeatureBuilder fit on ALL of train (everything before the
    project's split cutoff T) -- identical to Phases 1-2.
  - Training *examples* are drawn from the LATER part of train only (issues created after an
    inner cutoff T2 = 80th percentile of train's own `created`), and are scored using a
    FeatureBuilder fit on the EARLIER part of train only (created < T2). T2 precedes every
    training example's own creation time by construction, so no training example's features
    can see its own outcome or anything after it -- same rule as the test set, just applied
    one level up so the ranker has labeled examples to learn from at all.
  - The true assignee is force-included in TRAINING candidate sets (so every training group has
    a positive to learn from) -- this only affects which labels the supervised model gets to
    see, not what features it's allowed to use, so it isn't leakage. It is NOT done for TEST
    candidates, where only naturally-retrieved candidates are scored.
"""
import sys
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "/root/Issue-Assignee-Recommender/src")

from issue_assignee_recommender.data import PROJECTS, labeled, load_project
from issue_assignee_recommender.features import FEATURE_COLUMNS, FeatureBuilder
from issue_assignee_recommender.metrics import evaluate
from issue_assignee_recommender.training import build_rows


# Candidate widths matching the original (pre-2026-07-26) Phase 3 run: embeddings only, top-50
# neighbours, top-20 component owners, top-20 popular. Kept runnable via --legacy-candidates so
# the candidate-widening effect can be separated from the added features.
LEGACY_WIDTHS = dict(k_neighbors=50, bm25_k=0, comp_top=20, pop_top=20, recent_top=0)


def run_project(project_key: str, model: SentenceTransformer, widths: dict) -> dict:
    train, test = load_project(project_key)
    train_l = labeled(train)
    test_l = labeled(test)

    t2 = train["created"].quantile(0.8)
    hist_train_all = train[train["created"] < t2]
    hist_train_labeled = labeled(hist_train_all)
    train_examples = train_l[train_l["created"] >= t2]

    print(
        f"[{project_key}] history={len(hist_train_labeled)} train_examples={len(train_examples)} "
        f"full_train={len(train_l)} test={len(test_l)}",
        file=sys.stderr,
    )

    t0 = time.time()
    fb_train = FeatureBuilder(model, cutoff=t2, **widths)
    fb_train.fit(hist_train_labeled, hist_train_all)
    X_tr, y_tr, groups_tr, _ = build_rows(fb_train, train_examples, inject_true_label=True)
    print(
        f"  training rows={len(X_tr)} groups={len(groups_tr)} positives={y_tr.sum()} "
        f"({time.time()-t0:.1f}s)",
        file=sys.stderr,
    )

    t0 = time.time()
    fb_test = FeatureBuilder(model, cutoff=train["created"].max(), **widths)
    fb_test.fit(train_l, train)
    X_te, y_te, groups_te, meta_te = build_rows(fb_test, test_l, inject_true_label=False)
    print(f"  test rows={len(X_te)} groups={len(groups_te)} ({time.time()-t0:.1f}s)", file=sys.stderr)

    ranker = lgb.LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        n_estimators=200,
        num_leaves=31,
        learning_rate=0.05,
        min_child_samples=10,
        verbosity=-1,
    )
    ranker.fit(X_tr, y_tr, group=groups_tr)

    scores = ranker.predict(X_te)

    # reconstruct ranked candidate list per test issue
    predictions = []
    pos = 0
    idx_by_issue = {}
    for (issue_key, dev), s in zip(meta_te, scores):
        idx_by_issue.setdefault(issue_key, []).append((dev, s))

    truth_by_issue = dict(zip(test_l["issue_key"], test_l["assignee_key"]))
    # iterate over every test issue, not just those that produced candidates, so an issue with
    # an empty candidate set counts as a miss rather than quietly leaving the denominator
    for issue_key, true_dev in truth_by_issue.items():
        dev_scores = idx_by_issue.get(issue_key, [])
        ranked = [d for d, _ in sorted(dev_scores, key=lambda x: -x[1])]
        predictions.append((ranked, true_dev))

    metrics = evaluate(predictions, ks=(5, 10))
    # The ceiling the ranker is working against: no reordering can recover an issue whose true
    # assignee never entered the candidate set. Tracked as a first-class number because this,
    # not ranking quality, is what has been capping the metrics (see docs/phase3_ranking.md).
    oracle = sum(1 for ranked, true in predictions if true in ranked) / len(predictions)
    metrics["oracle_recall"] = oracle
    metrics["ndcg@10_over_oracle"] = metrics["ndcg@10"] / oracle if oracle else 0.0
    print(f"  LightGBM ranker: {metrics}", file=sys.stderr)

    importances = dict(zip(FEATURE_COLUMNS, ranker.feature_importances_.tolist()))
    print(f"  feature importances: {importances}", file=sys.stderr)

    return {"project": project_key, "method": "lightgbm_lambdarank", **metrics, **{
        f"importance_{k}": v for k, v in importances.items()
    }}


def main():
    legacy = "--legacy-candidates" in sys.argv
    widths = LEGACY_WIDTHS if legacy else {}
    out_path = (
        "/root/Issue-Assignee-Recommender/data/processed/"
        + ("phase3_results_legacy_candidates.csv" if legacy else "phase3_results.csv")
    )
    print(f"candidate widths: {widths or 'defaults (features.py)'}", file=sys.stderr)
    print("loading embedding model ...", file=sys.stderr)
    model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2",
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    rows = []
    for project_key in PROJECTS:
        rows.append(run_project(project_key, model, widths))

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(
        "\n"
        + df[
            ["project", "recall@5", "recall@10", "mrr", "ndcg@10", "oracle_recall"]
        ].to_string(index=False)
    )
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
