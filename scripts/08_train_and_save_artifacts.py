"""Phase 6 prep: train the Phase 3 pipeline once per project and persist artifacts to disk
(FAISS index + feature-builder state + LightGBM model) so the API server doesn't have to
retrain at startup. Uses the SAME full-train FeatureBuilder as Phase 3's test scoring (fit on
all train-period issues, cutoff = each project's split date T) -- this is the artifact a real
"score a brand new issue right now" deployment would use, so training examples are drawn the
same way Phase 3 did (inner T2 split) purely to fit the ranker.
"""
import sys
import time

import lightgbm as lgb
from sentence_transformers import SentenceTransformer

sys.path.insert(0, "/root/Issue-Assignee-Recommender/src")

from issue_assignee_recommender.data import PROJECTS, labeled, load_project
from issue_assignee_recommender.features import FeatureBuilder
from issue_assignee_recommender.training import build_rows

MODEL_DIR = "/root/Issue-Assignee-Recommender/data/models"


def main():
    import os

    os.makedirs(MODEL_DIR, exist_ok=True)
    print("loading embedding model ...", file=sys.stderr)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")

    for project_key in PROJECTS:
        t0 = time.time()
        train, _test = load_project(project_key)
        train_l = labeled(train)

        t2 = train["created"].quantile(0.8)
        hist_train_all = train[train["created"] < t2]
        hist_train_labeled = labeled(hist_train_all)
        train_examples = train_l[train_l["created"] >= t2]

        fb_train = FeatureBuilder(model, cutoff=t2)
        fb_train.fit(hist_train_labeled, hist_train_all)
        X_tr, y_tr, groups_tr, _ = build_rows(fb_train, train_examples, inject_true_label=True)

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

        # the deployment-facing FeatureBuilder: fit on ALL train (matches Phase 3 test scoring)
        fb_serve = FeatureBuilder(model, cutoff=train["created"].max())
        fb_serve.fit(train_l, train)

        prefix = f"{MODEL_DIR}/{project_key}"
        fb_serve.save(prefix)
        ranker.booster_.save_model(f"{prefix}_lgbm.txt")

        print(f"[{project_key}] saved artifacts to {prefix}* ({time.time()-t0:.1f}s)", file=sys.stderr)


if __name__ == "__main__":
    main()
