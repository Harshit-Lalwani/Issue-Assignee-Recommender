"""Shared helper for turning (issue, FeatureBuilder) pairs into LightGBM-ready rows."""
import numpy as np

from .features import FEATURE_COLUMNS

ISSUE_FIELDS = ("summary", "description", "components", "reporter_key", "issuetype")


def build_rows(fb, issues_df, inject_true_label: bool):
    """Returns (X, y, groups, meta) where meta has (issue_key, dev) per row, for later
    reconstruction of ranked lists per issue. Candidate-source widths come from the
    FeatureBuilder's own configuration -- see features.py."""
    X, y, groups, meta = [], [], [], []
    for row in issues_df.itertuples():
        issue_row = {f: getattr(row, f, None) for f in ISSUE_FIELDS}
        ctx = fb.score_issue(issue_row)
        cand = ctx["candidates"]
        true_dev = row.assignee_key
        if inject_true_label:
            cand.add(true_dev)
        if not cand:
            continue
        rows_for_issue = 0
        for dev in cand:
            feats = fb.feature_row(issue_row, dev, ctx)
            X.append([feats[c] for c in FEATURE_COLUMNS])
            y.append(1 if dev == true_dev else 0)
            meta.append((row.issue_key, dev))
            rows_for_issue += 1
        groups.append(rows_for_issue)
    return np.array(X, dtype="float64"), np.array(y, dtype="int32"), groups, meta
