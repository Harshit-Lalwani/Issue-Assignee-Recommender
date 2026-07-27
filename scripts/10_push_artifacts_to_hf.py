"""Phase 6 deployment: push trained artifacts (data/models/*) to a public Hugging Face Hub
model repo, so a fresh clone/deploy of this project (where data/models/ is gitignored, ~107MB
of binary artifacts not sensibly tracked in git) can fetch them at startup instead of requiring
a local training run first. Counterpart to the download logic in
src/issue_assignee_recommender/api.py (_ensure_artifacts_local).

Requires HF_TOKEN in .env with write access to the target repo (create one at
https://huggingface.co/settings/tokens).
"""
import sys

from huggingface_hub import HfApi, create_repo

sys.path.insert(0, "/root/Issue-Assignee-Recommender/src")
from issue_assignee_recommender.llm import load_env  # reuses the existing .env parser

REPO_ID = "MegaKnight9x/issue-assignee-recommender-artifacts"
MODEL_DIR = "/root/Issue-Assignee-Recommender/data/models"


def main():
    env = load_env()
    token = env.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN missing from .env", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)
    create_repo(REPO_ID, token=token, repo_type="model", private=False, exist_ok=True)
    print(f"Uploading {MODEL_DIR} -> https://huggingface.co/{REPO_ID}", file=sys.stderr)

    api.upload_folder(
        folder_path=MODEL_DIR,
        repo_id=REPO_ID,
        repo_type="model",
        commit_message="Upload trained per-project artifacts (FAISS index, feature-builder state, LightGBM model)",
    )
    print("Done.", file=sys.stderr)


if __name__ == "__main__":
    main()
