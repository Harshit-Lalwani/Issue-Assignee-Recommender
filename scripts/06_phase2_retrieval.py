"""Phase 2 deliverable: Recall@50 of the true assignee, embeddings vs BM25.

Both retrievers aggregate resolvers over their top-50 nearest train issues (by BM25 score or
cosine similarity), padding with global popularity if fewer than 50 distinct resolvers turn up
-- same methodology for both, so the comparison isolates text-similarity quality, not padding
behavior.
"""
import sys
import time

sys.path.insert(0, "/root/Issue-Assignee-Recommender/src")

from sentence_transformers import SentenceTransformer

from issue_assignee_recommender.baselines import BM25Baseline
from issue_assignee_recommender.data import PROJECTS, labeled, load_project
from issue_assignee_recommender.metrics import recall_at_k
from issue_assignee_recommender.retrieval import MODEL_NAME, EmbeddingRetriever


def run_project(project_key: str, model: SentenceTransformer) -> list[dict]:
    train, test = load_project(project_key)
    train_l = labeled(train)
    test_l = labeled(test)
    print(f"[{project_key}] train={len(train_l)} test={len(test_l)}", file=sys.stderr)

    rows = []

    t0 = time.time()
    bm25 = BM25Baseline(k_neighbors=50)
    bm25.fit(train_l)
    bm25_preds = []
    for row in test_l.itertuples():
        row_d = {"summary": row.summary, "description": row.description}
        bm25_preds.append((bm25.rank(row_d, top_n=50), row.assignee_key))
    bm25_recall50 = sum(recall_at_k(r, t, 50) for r, t in bm25_preds) / len(bm25_preds)
    print(f"  bm25 recall@50={bm25_recall50:.4f} ({time.time()-t0:.1f}s)", file=sys.stderr)
    rows.append({"project": project_key, "method": "bm25_knn", "recall@50": bm25_recall50})

    t0 = time.time()
    emb = EmbeddingRetriever(model, k_neighbors=50)
    emb.fit(train_l)
    ranked_lists = emb.rank_batch(test_l, top_n=50)
    truths = test_l["assignee_key"].tolist()
    emb_recall50 = sum(recall_at_k(r, t, 50) for r, t in zip(ranked_lists, truths)) / len(truths)
    print(f"  embedding recall@50={emb_recall50:.4f} ({time.time()-t0:.1f}s)", file=sys.stderr)
    rows.append({"project": project_key, "method": "embedding_faiss", "recall@50": emb_recall50})

    return rows


def main():
    print(f"loading {MODEL_NAME} ...", file=sys.stderr)
    model = SentenceTransformer(MODEL_NAME, device="cuda")

    all_rows = []
    for project_key in PROJECTS:
        all_rows.extend(run_project(project_key, model))

    import pandas as pd

    df = pd.DataFrame(all_rows)
    out_path = "/root/Issue-Assignee-Recommender/data/processed/phase2_recall50.csv"
    df.to_csv(out_path, index=False)
    print("\n" + df.pivot(index="project", columns="method", values="recall@50").to_string())
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
