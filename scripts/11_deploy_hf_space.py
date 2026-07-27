"""Phase 6 deployment: create (or update) the public Hugging Face Space and push everything
the Docker build needs. Idempotent -- safe to rerun after any code change to redeploy.
"""
import sys

from huggingface_hub import CommitOperationAdd, HfApi, create_repo

sys.path.insert(0, "/root/Issue-Assignee-Recommender/src")
from issue_assignee_recommender.llm import load_env

SPACE_ID = "MegaKnight9x/issue-assignee-recommender"
ROOT = "/root/Issue-Assignee-Recommender"


def main():
    env = load_env()
    token = env.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN missing from .env", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)
    create_repo(SPACE_ID, token=token, repo_type="space", space_sdk="docker", private=False, exist_ok=True)
    print(f"Space: https://huggingface.co/spaces/{SPACE_ID}", file=sys.stderr)

    ops = [
        CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=f"{ROOT}/space_README.md"),
        CommitOperationAdd(path_in_repo="Dockerfile", path_or_fileobj=f"{ROOT}/Dockerfile"),
        CommitOperationAdd(
            path_in_repo="requirements-serve.txt", path_or_fileobj=f"{ROOT}/requirements-serve.txt"
        ),
        CommitOperationAdd(
            path_in_repo="data/processed/phase3_results.csv",
            path_or_fileobj=f"{ROOT}/data/processed/phase3_results.csv",
        ),
        CommitOperationAdd(
            path_in_repo="data/processed/split_manifest.csv",
            path_or_fileobj=f"{ROOT}/data/processed/split_manifest.csv",
        ),
    ]

    import os

    src_root = f"{ROOT}/src"
    for dirpath, _dirnames, filenames in os.walk(src_root):
        for fname in filenames:
            if fname.endswith(".pyc") or "__pycache__" in dirpath:
                continue
            local_path = os.path.join(dirpath, fname)
            rel_path = "src/" + os.path.relpath(local_path, src_root)
            ops.append(CommitOperationAdd(path_in_repo=rel_path, path_or_fileobj=local_path))

    print(f"Pushing {len(ops)} files ...", file=sys.stderr)
    api.create_commit(
        repo_id=SPACE_ID,
        repo_type="space",
        operations=ops,
        commit_message="Deploy: fix CPU device selection, HF Hub artifact fetch, Docker build",
    )
    print("Pushed. Build will start automatically.", file=sys.stderr)


if __name__ == "__main__":
    main()
