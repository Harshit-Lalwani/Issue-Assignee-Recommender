"""Phase 6: minimal serving API.

Loads pretrained artifacts (one FeatureBuilder + LightGBM model per project, produced by
scripts/08_train_and_save_artifacts.py) once at startup, and the shared sentence-transformer
encoder once (identical across projects -- no reason to duplicate it in memory 5x).

POST /recommend/{project} -- rank developers for a not-yet-existing issue, exactly as if it
had just been filed. Only fields legitimately available at creation time are accepted, per
docs/phase0_framing.md.
"""
import time
from pathlib import Path

import lightgbm as lgb
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

from .data import PROJECTS
from .features import FeatureBuilder

MODEL_DIR = Path(__file__).resolve().parents[2] / "data" / "models"
RESULTS_CSV = Path(__file__).resolve().parents[2] / "data" / "processed" / "phase3_results.csv"

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


@app.on_event("startup")
def load_artifacts():
    _state["encoder"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    for project in PROJECTS:
        prefix = str(MODEL_DIR / project)
        if not Path(f"{prefix}.pkl").exists():
            continue
        _state["builders"][project] = FeatureBuilder.load(prefix, _state["encoder"])
        _state["rankers"][project] = lgb.Booster(model_file=f"{prefix}_lgbm.txt")
    if RESULTS_CSV.exists():
        _state["metrics"] = pd.read_csv(RESULTS_CSV)


@app.get("/health")
def health():
    return {"status": "ok", "projects_loaded": list(_state["builders"].keys())}


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

    from .features import FEATURE_COLUMNS

    devs = list(cand)
    X = [[fb.feature_row(issue_row, d, ctx)[c] for c in FEATURE_COLUMNS] for d in devs]
    scores = ranker.predict(X)
    ranked = sorted(zip(devs, scores), key=lambda x: -x[1])[: issue.top_n]

    return RecommendOut(
        project=project,
        candidates=[Candidate(developer_id=d, score=float(s)) for d, s in ranked],
        latency_ms=(time.time() - t0) * 1000,
    )
