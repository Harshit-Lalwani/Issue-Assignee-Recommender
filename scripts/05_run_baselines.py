"""Phase 1 deliverable: leaderboard of Recall@5, Recall@10, MRR, NDCG@10 for the three
required baselines (popularity, component-owner, BM25 kNN), per project.

Evaluated strictly on the labeled subset of each project's test split (issues that do have a
resolved assignee -- see docs/phase0_framing.md for why this is a much smaller slice than the
raw test split for JRASERVER specifically).
"""
import sys
import time

sys.path.insert(0, "/root/Issue-Assignee-Recommender/src")

from issue_assignee_recommender.baselines import (
    BM25Baseline,
    ComponentOwnerBaseline,
    PopularityBaseline,
)
from issue_assignee_recommender.data import PROJECTS, labeled, load_project
from issue_assignee_recommender.metrics import evaluate, leaderboard

METHODS = [
    lambda: PopularityBaseline(),
    lambda: ComponentOwnerBaseline(),
    lambda: BM25Baseline(k_neighbors=50),
]


def run_project(project_key: str) -> list[dict]:
    train, test = load_project(project_key)
    train_l = labeled(train)
    test_l = labeled(test)
    print(
        f"[{project_key}] train_labeled={len(train_l)} test_labeled={len(test_l)}",
        file=sys.stderr,
    )

    rows = []
    for make_method in METHODS:
        method = make_method()
        t0 = time.time()
        method.fit(train_l)
        predictions = []
        for row in test_l.itertuples():
            row_d = {
                "summary": row.summary,
                "description": row.description,
                "components": row.components,
            }
            ranked = method.rank(row_d, top_n=10)
            predictions.append((ranked, row.assignee_key))
        metrics = evaluate(predictions, ks=(5, 10))
        elapsed = time.time() - t0
        print(
            f"  {method.name}: {metrics} ({elapsed:.1f}s)",
            file=sys.stderr,
        )
        rows.append({"project": project_key, "method": method.name, **metrics})
    return rows


def main():
    all_rows = []
    for project_key in PROJECTS:
        all_rows.extend(run_project(project_key))

    lb = leaderboard(all_rows)
    out_path = "/root/Issue-Assignee-Recommender/data/processed/phase1_leaderboard.csv"
    lb.to_csv(out_path, index=False)
    print("\n" + lb.to_string(index=False))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
