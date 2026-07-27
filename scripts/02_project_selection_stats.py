"""
Phase 0 project selection: compute, per (instance, project) pair, the stats needed to
pick 3-5 sub-projects with genuinely distributed contribution (not dominated by 1-2 devs).

Runs an aggregation directly against the restored MongoDB collections so we don't have to
export all 2.7M issues before knowing which ~5 projects we actually want.
"""
import sys
from collections import defaultdict

import pandas as pd
from pymongo import MongoClient

MONGO_URI = "mongodb://127.0.0.1:27017"
DB_NAME = "JiraRepos"

# Instances worth considering: Apache (huge, diverse sub-projects), Jira and JiraEcosystem
# (Atlassian's own trackers -- maximally on-domain), Spring, MongoDB, RedHat, Qt as backups.
CANDIDATE_INSTANCES = ["Apache", "Jira", "JiraEcosystem", "Spring", "MongoDB", "RedHat", "Qt"]


def project_stats_for_instance(db, instance: str) -> pd.DataFrame:
    coll = db[instance]
    pipeline = [
        {
            "$project": {
                "project_key": "$fields.project.key",
                "created": "$fields.created",
                "resolutiondate": "$fields.resolutiondate",
                "assignee_key": "$fields.assignee.key",
            }
        },
        {
            "$group": {
                "_id": "$project_key",
                "issue_count": {"$sum": 1},
                "resolved_count": {
                    "$sum": {"$cond": [{"$ne": ["$resolutiondate", None]}, 1, 0]}
                },
                "min_created": {"$min": "$created"},
                "max_created": {"$max": "$created"},
            }
        },
    ]
    rows = list(coll.aggregate(pipeline, allowDiskUse=True))
    df = pd.DataFrame(rows).rename(columns={"_id": "project_key"})
    df["instance"] = instance
    return df


def resolver_concentration(db, instance: str, project_key: str) -> dict:
    """Top-2-developer share of RESOLVED issues for one project, plus distinct resolver count."""
    coll = db[instance]
    pipeline = [
        {
            "$match": {
                "fields.project.key": project_key,
                "fields.resolutiondate": {"$ne": None},
                "fields.assignee.key": {"$ne": None},
            }
        },
        {"$group": {"_id": "$fields.assignee.key", "n": {"$sum": 1}}},
        {"$sort": {"n": -1}},
    ]
    counts = [r["n"] for r in coll.aggregate(pipeline, allowDiskUse=True)]
    total = sum(counts)
    if total == 0:
        return {"resolved_with_assignee": 0, "distinct_resolvers": 0, "top2_share": None}
    top2 = sum(counts[:2])
    return {
        "resolved_with_assignee": total,
        "distinct_resolvers": len(counts),
        "top2_share": round(top2 / total, 3),
    }


def main():
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    available = set(db.list_collection_names())

    all_rows = []
    for instance in CANDIDATE_INSTANCES:
        if instance not in available:
            print(f"skip {instance}: not in restored dump", file=sys.stderr)
            continue
        print(f"aggregating {instance} ...", file=sys.stderr)
        df = project_stats_for_instance(db, instance)
        all_rows.append(df)

    stats = pd.concat(all_rows, ignore_index=True)
    stats = stats.sort_values("issue_count", ascending=False)

    # Only bother computing resolver concentration for projects big enough to matter
    # (>= 500 issues), otherwise Recall@k there is too noisy regardless of concentration.
    big = stats[stats["issue_count"] >= 500].copy()
    print(f"Computing resolver concentration for {len(big)} candidate projects ...", file=sys.stderr)

    conc_rows = []
    for _, row in big.iterrows():
        conc = resolver_concentration(db, row["instance"], row["project_key"])
        conc_rows.append({**row.to_dict(), **conc})

    result = pd.DataFrame(conc_rows)
    result = result.sort_values("issue_count", ascending=False)

    out_path = "/root/Issue-Assignee-Recommender/data/processed/project_selection_stats.csv"
    result.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(result)} rows)", file=sys.stderr)

    pd.set_option("display.max_rows", 200)
    pd.set_option("display.width", 200)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
