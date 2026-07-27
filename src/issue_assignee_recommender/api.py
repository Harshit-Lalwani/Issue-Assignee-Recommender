"""Phase 6: minimal serving API.

Loads pretrained artifacts (one FeatureBuilder + LightGBM model per project, produced by
scripts/08_train_and_save_artifacts.py) once at startup, and the shared sentence-transformer
encoder once (identical across projects -- no reason to duplicate it in memory 5x).

Artifacts are ~107MB, too large/binary to sensibly track in the GitHub repo (see
scripts/10_push_artifacts_to_hf.py) -- they live in a public Hugging Face Hub model repo and
are downloaded into data/models/ at startup if not already present locally, which is what makes
this runnable from a plain `git clone` with no local training step.

POST /recommend/{project} -- rank developers for a not-yet-existing issue, exactly as if it
had just been filed. Only fields legitimately available at creation time are accepted, per
docs/phase0_framing.md.
"""
import os
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from huggingface_hub import hf_hub_download
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from .data import PROJECTS
from .features import FEATURE_COLUMNS, FeatureBuilder

MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
RESULTS_CSV = Path(__file__).resolve().parents[2] / "data" / "processed" / "phase3_results.csv"

# Public HF Hub model repo holding the trained artifacts (scripts/10_push_artifacts_to_hf.py).
# Override via env var so a fork/clone can point at their own copy without touching code.
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "MegaKnight9x/issue-assignee-recommender-artifacts")
ARTIFACT_SUFFIXES = (".faiss", ".pkl", "_lgbm.txt")

app = FastAPI(
    title="Issue-Assignee Recommender",
    description="Ranks the developers most likely to resolve a newly-filed Jira issue.",
)

_state: dict = {"encoder": None, "builders": {}, "rankers": {}, "metrics": None}


class IssueIn(BaseModel):
    summary: str = ""
    description: str = ""
    components: list[str] = []
    reporter_key: str | None = None
    issuetype: str | None = None
    top_n: int = 5


class Candidate(BaseModel):
    developer_id: str
    score: float


class RecommendOut(BaseModel):
    project: str
    candidates: list[Candidate]
    latency_ms: float


def _ensure_artifacts_local(project: str) -> bool:
    """Download this project's 3 artifact files from the HF Hub model repo if they aren't
    already sitting in data/models/ (e.g. from a local training run). Returns False if the
    files aren't available anywhere -- caller skips that project rather than crashing."""
    prefix = MODEL_DIR / project
    if all(Path(f"{prefix}{suf}").exists() for suf in ARTIFACT_SUFFIXES):
        return True
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        for suf in ARTIFACT_SUFFIXES:
            fname = f"{project}{suf}"
            local_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=fname)
            dest = Path(f"{prefix}{suf}")
            if not dest.exists():
                dest.symlink_to(local_path)
        return True
    except Exception as e:  # noqa: BLE001 -- best-effort; missing project just gets skipped
        print(f"[startup] could not fetch artifacts for {project}: {e}")
        return False


@app.on_event("startup")
def load_artifacts():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
    _state["encoder"] = encoder
    _state["device"] = device

    for project in PROJECTS:
        if not _ensure_artifacts_local(project):
            continue
        prefix = str(MODEL_DIR / project)
        _state["builders"][project] = FeatureBuilder.load(prefix, encoder)
        _state["rankers"][project] = lgb.Booster(model_file=f"{prefix}_lgbm.txt")

    if RESULTS_CSV.exists():
        _state["metrics"] = pd.read_csv(RESULTS_CSV)

    # Warm the encoder + a real candidate/feature pass so the first real request doesn't eat
    # the lazy-init cost (first call into a fresh SentenceTransformer/torch graph is slow).
    if _state["builders"]:
        warm_project = next(iter(_state["builders"]))
        try:
            recommend(warm_project, IssueIn(summary="warmup", description="warmup request"))
        except Exception as e:  # noqa: BLE001 -- warmup is best-effort, never block startup
            print(f"[startup] warmup call failed (non-fatal): {e}")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": _state.get("device"),
        "projects_loaded": list(_state["builders"].keys()),
    }


@app.get("/projects")
def projects():
    if _state["metrics"] is None:
        return {"projects": list(_state["builders"].keys())}
    return _state["metrics"][["project", "recall@5", "recall@10", "mrr", "ndcg@10"]].to_dict(
        orient="records"
    )


@app.post("/recommend/{project}", response_model=RecommendOut)
def recommend(project: str, issue: IssueIn):
    if project not in _state["builders"]:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown project '{project}'. Available: {list(_state['builders'].keys())}",
        )
    t0 = time.time()
    fb: FeatureBuilder = _state["builders"][project]
    ranker: lgb.Booster = _state["rankers"][project]

    issue_row = {
        "summary": issue.summary,
        "description": issue.description,
        "components": issue.components,
        "reporter_key": issue.reporter_key,
        "issuetype": issue.issuetype,
    }
    ctx = fb.score_issue(issue_row)
    cand = ctx["candidates"]
    if not cand:
        return RecommendOut(project=project, candidates=[], latency_ms=(time.time() - t0) * 1000)

    devs = list(cand)
    X = [[fb.feature_row(issue_row, d, ctx)[c] for c in FEATURE_COLUMNS] for d in devs]
    scores = ranker.predict(X)
    ranked = sorted(zip(devs, scores), key=lambda x: -x[1])[: issue.top_n]

    return RecommendOut(
        project=project,
        candidates=[Candidate(developer_id=d, score=float(s)) for d, s in ranked],
        latency_ms=(time.time() - t0) * 1000,
    )


@app.get("/", response_class=HTMLResponse)
def index():
    projects_loaded = list(_state["builders"].keys()) or PROJECTS
    options = "\n".join(f'<option value="{p}">{p}</option>' for p in projects_loaded)
    return f"""
<!doctype html>
<html>
<head>
<title>Issue-Assignee Recommender</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 720px; margin: 40px auto; padding: 0 16px; color: #1a1a1a; }}
  h1 {{ font-size: 1.4rem; }}
  textarea, select, input {{ width: 100%; margin: 6px 0 14px; padding: 8px; font-size: 1rem; box-sizing: border-box; }}
  button {{ padding: 10px 18px; font-size: 1rem; cursor: pointer; }}
  #out {{ white-space: pre-wrap; background: #f5f5f5; padding: 12px; border-radius: 6px; margin-top: 16px; font-family: monospace; font-size: 0.85rem; }}
  a {{ color: #0645ad; }}
</style>
</head>
<body>
<h1>Issue-Assignee Recommender</h1>
<p>Two-stage recommender (BM25/embedding retrieval &rarr; LightGBM LambdaRank) that ranks the
developers most likely to resolve a newly-filed Jira issue, trained on 129K real issues from
5 open-source projects. Source + full writeup:
<a href="https://github.com/Harshit-Lalwani/Issue-Assignee-Recommender">GitHub</a>.</p>

<label>Project</label>
<select id="project">{options}</select>

<label>Issue summary</label>
<input id="summary" value="Consumer group rebalance causes duplicate message processing" />

<label>Issue description</label>
<textarea id="description" rows="3">When a consumer in the group restarts, duplicate processing occurs around the rebalance window.</textarea>

<button onclick="submitIssue()">Recommend developers</button>
<div id="out"></div>

<script>
async function submitIssue() {{
  const project = document.getElementById('project').value;
  const summary = document.getElementById('summary').value;
  const description = document.getElementById('description').value;
  document.getElementById('out').textContent = 'Scoring...';
  const t0 = performance.now();
  const res = await fetch(`/recommend/${{project}}`, {{
    method: 'POST',
    headers: {{'Content-Type': 'application/json'}},
    body: JSON.stringify({{summary, description, top_n: 5}})
  }});
  const data = await res.json();
  const clientMs = (performance.now() - t0).toFixed(0);
  document.getElementById('out').textContent =
    JSON.stringify(data, null, 2) + `\\n\\n(round-trip incl. network: ${{clientMs}}ms)`;
}}
</script>
</body>
</html>
"""
