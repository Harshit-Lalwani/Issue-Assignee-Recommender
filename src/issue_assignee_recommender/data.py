"""Loading helpers for the frozen Phase 0 export."""
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

PROJECTS = ["JRASERVER", "SPARK", "HADOOP", "KAFKA", "CASSANDRA"]


def load_issues_split() -> pd.DataFrame:
    """All issues (train+test) with the frozen per-project 80th-pct split column."""
    return pd.read_parquet(DATA_DIR / "issues_split.parquet")


def load_project(project_key: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) for one project, unfiltered on label presence."""
    df = load_issues_split()
    df = df[df["project_key"] == project_key]
    train = df[df["split"] == "train"].copy()
    test = df[df["split"] == "test"].copy()
    return train, test


def labeled(df: pd.DataFrame) -> pd.DataFrame:
    """Issues with a real, resolved assignee -- the only ones usable as ground truth,
    or as training signal for 'who resolved similar issues before'."""
    return df[df["assignee_key"].notna() & df["resolutiondate"].notna()].copy()


def load_changelog() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "changelog.parquet")


def load_comments() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "comments.parquet")


def issue_text(row) -> str:
    """Text available at issue-creation time -- summary + description only."""
    summary = row.get("summary") or ""
    description = row.get("description") or ""
    return f"{summary}\n{description}"
