"""
Phase 0: export the 5 selected projects from MongoDB to Parquet.

Produces, under data/processed/:
  issues.parquet      -- one row per issue (text + creation-time metadata + resolution outcome)
  changelog.parquet   -- one row per changelog history entry (author, timestamp, changed fields)
  comments.parquet    -- one row per comment (author, timestamp only -- bodies dropped, not
                         needed for retrieval/ranking and they bloat storage substantially)

Only fields that exist at or are needed to compute leakage-free training data are kept.
`assignee_key` and `resolutiondate` are the prediction targets / outcome fields -- consumers
must not treat them as inputs for a live issue, only as labels for historical ones.
"""
import sys
from datetime import datetime, timezone

import pyarrow as pa
import pyarrow.parquet as pq
from pymongo import MongoClient

MONGO_URI = "mongodb://127.0.0.1:27017"
DB_NAME = "JiraRepos"
OUT_DIR = "/root/Issue-Assignee-Recommender/data/processed"

# (instance, project_key)
SELECTED_PROJECTS = [
    ("Jira", "JRASERVER"),
    ("Apache", "SPARK"),
    ("Apache", "HADOOP"),
    ("Apache", "KAFKA"),
    ("Apache", "CASSANDRA"),
]

# Dataset was collected Jan 2022; anything outside this range is a bad/anomalous timestamp
# (e.g. one SPARK issue has created="0010-04-03"). Drop rather than silently mis-sort.
SANE_MIN = datetime(1995, 1, 1, tzinfo=timezone.utc)
SANE_MAX = datetime(2022, 2, 1, tzinfo=timezone.utc)


def parse_dt(s):
    if not s:
        return None
    try:
        # Jira timestamps: "2022-01-04T20:45:19.000+0000" or with "-0600" style offsets
        dt = datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S")
        # attach offset roughly (we only need this for sane-range filtering + sorting,
        # not sub-minute precision comparisons across offsets)
        return dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def author_key(obj):
    if not obj:
        return None
    return obj.get("key")


def export_project(db, instance, project_key, issues_writer, changelog_writer, comments_writer):
    coll = db[instance]
    cursor = coll.find({"fields.project.key": project_key}, no_cursor_timeout=True)

    issue_rows = []
    changelog_rows = []
    comment_rows = []
    n = 0
    n_dropped_bad_date = 0

    for doc in cursor:
        f = doc.get("fields", {})
        created = parse_dt(f.get("created"))
        if created is None or created < SANE_MIN or created > SANE_MAX:
            n_dropped_bad_date += 1
            continue

        resolutiondate = parse_dt(f.get("resolutiondate"))
        components = [c.get("name") for c in (f.get("components") or []) if c.get("name")]
        labels = f.get("labels") or []
        issuelinks = f.get("issuelinks") or []

        issue_rows.append(
            {
                "instance": instance,
                "project_key": project_key,
                "issue_key": doc.get("key"),
                "issuetype": (f.get("issuetype") or {}).get("name"),
                "priority": (f.get("priority") or {}).get("name") if f.get("priority") else None,
                "status": (f.get("status") or {}).get("name"),
                "summary": f.get("summary"),
                "description": f.get("description"),
                "components": components,
                "labels": labels,
                "num_issuelinks_at_export": len(issuelinks),
                "created": created,
                "resolutiondate": resolutiondate,
                "resolution": (f.get("resolution") or {}).get("name") if f.get("resolution") else None,
                "reporter_key": author_key(f.get("reporter")),
                "creator_key": author_key(f.get("creator")),
                "assignee_key": author_key(f.get("assignee")),
            }
        )

        for h in (doc.get("changelog", {}) or {}).get("histories", []) or []:
            h_created = parse_dt(h.get("created"))
            ak = author_key(h.get("author"))
            for item in h.get("items", []) or []:
                changelog_rows.append(
                    {
                        "instance": instance,
                        "project_key": project_key,
                        "issue_key": doc.get("key"),
                        "history_id": h.get("id"),
                        "author_key": ak,
                        "created": h_created,
                        "field": item.get("field"),
                        "from_value": item.get("fromString"),
                        "to_value": item.get("toString"),
                    }
                )

        for c in f.get("comments", []) or []:
            comment_rows.append(
                {
                    "instance": instance,
                    "project_key": project_key,
                    "issue_key": doc.get("key"),
                    "comment_id": c.get("id"),
                    "author_key": author_key(c.get("author")),
                    "created": parse_dt(c.get("created")),
                }
            )

        n += 1
        if n % 5000 == 0:
            print(f"  {instance}/{project_key}: {n} issues processed...", file=sys.stderr)
            issues_writer.write_table(pa.Table.from_pylist(issue_rows, schema=ISSUE_SCHEMA))
            issue_rows = []
            if changelog_rows:
                changelog_writer.write_table(pa.Table.from_pylist(changelog_rows, schema=CHANGELOG_SCHEMA))
                changelog_rows = []
            if comment_rows:
                comments_writer.write_table(pa.Table.from_pylist(comment_rows, schema=COMMENT_SCHEMA))
                comment_rows = []

    if issue_rows:
        issues_writer.write_table(pa.Table.from_pylist(issue_rows, schema=ISSUE_SCHEMA))
    if changelog_rows:
        changelog_writer.write_table(pa.Table.from_pylist(changelog_rows, schema=CHANGELOG_SCHEMA))
    if comment_rows:
        comments_writer.write_table(pa.Table.from_pylist(comment_rows, schema=COMMENT_SCHEMA))

    cursor.close()
    print(
        f"done {instance}/{project_key}: {n} issues exported, {n_dropped_bad_date} dropped (bad created date)",
        file=sys.stderr,
    )
    return n, n_dropped_bad_date


ISSUE_SCHEMA = pa.schema(
    [
        ("instance", pa.string()),
        ("project_key", pa.string()),
        ("issue_key", pa.string()),
        ("issuetype", pa.string()),
        ("priority", pa.string()),
        ("status", pa.string()),
        ("summary", pa.string()),
        ("description", pa.string()),
        ("components", pa.list_(pa.string())),
        ("labels", pa.list_(pa.string())),
        ("num_issuelinks_at_export", pa.int32()),
        ("created", pa.timestamp("us", tz="UTC")),
        ("resolutiondate", pa.timestamp("us", tz="UTC")),
        ("resolution", pa.string()),
        ("reporter_key", pa.string()),
        ("creator_key", pa.string()),
        ("assignee_key", pa.string()),
    ]
)
CHANGELOG_SCHEMA = pa.schema(
    [
        ("instance", pa.string()),
        ("project_key", pa.string()),
        ("issue_key", pa.string()),
        ("history_id", pa.string()),
        ("author_key", pa.string()),
        ("created", pa.timestamp("us", tz="UTC")),
        ("field", pa.string()),
        ("from_value", pa.string()),
        ("to_value", pa.string()),
    ]
)
COMMENT_SCHEMA = pa.schema(
    [
        ("instance", pa.string()),
        ("project_key", pa.string()),
        ("issue_key", pa.string()),
        ("comment_id", pa.string()),
        ("author_key", pa.string()),
        ("created", pa.timestamp("us", tz="UTC")),
    ]
)


def main():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]

    summary = []
    with pq.ParquetWriter(f"{OUT_DIR}/issues.parquet", ISSUE_SCHEMA) as iw, pq.ParquetWriter(
        f"{OUT_DIR}/changelog.parquet", CHANGELOG_SCHEMA
    ) as cw, pq.ParquetWriter(f"{OUT_DIR}/comments.parquet", COMMENT_SCHEMA) as mw:
        for instance, project_key in SELECTED_PROJECTS:
            print(f"exporting {instance}/{project_key} ...", file=sys.stderr)
            n, dropped = export_project(db, instance, project_key, iw, cw, mw)
            summary.append((instance, project_key, n, dropped))

    print("\nSummary:")
    for instance, project_key, n, dropped in summary:
        print(f"  {instance}/{project_key}: {n} issues ({dropped} dropped)")


if __name__ == "__main__":
    main()
