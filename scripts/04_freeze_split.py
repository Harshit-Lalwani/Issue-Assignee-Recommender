"""
Phase 0 deliverable: freeze the train/test temporal split.

Per project, cutoff T = 80th percentile of `created`. Train = created < T, test = created >= T.
Writes data/processed/split_manifest.csv (one row per project: T, train/test counts) and adds
a `split` column ('train'/'test') to a copy of issues at data/processed/issues_split.parquet.
"""
import duckdb

CON = duckdb.connect()
ISSUES = "/root/Issue-Assignee-Recommender/data/processed/issues.parquet"
OUT_MANIFEST = "/root/Issue-Assignee-Recommender/data/processed/split_manifest.csv"
OUT_ISSUES = "/root/Issue-Assignee-Recommender/data/processed/issues_split.parquet"

projects = CON.execute(f"SELECT DISTINCT project_key FROM read_parquet('{ISSUES}')").fetchall()
projects = [p[0] for p in projects]

cutoffs = {}
for p in projects:
    t = CON.execute(
        f"""
        SELECT quantile_cont(epoch(created), 0.8)
        FROM read_parquet('{ISSUES}')
        WHERE project_key = ?
        """,
        [p],
    ).fetchone()[0]
    cutoffs[p] = t

case_expr = " ".join(
    f"WHEN project_key = '{p}' THEN to_timestamp({t})" for p, t in cutoffs.items()
)

CON.execute(
    f"""
    COPY (
        SELECT *,
            CASE {case_expr} END AS split_cutoff,
            CASE
                WHEN created < (CASE {case_expr} END) THEN 'train'
                ELSE 'test'
            END AS split
        FROM read_parquet('{ISSUES}')
    ) TO '{OUT_ISSUES}' (FORMAT PARQUET)
    """
)

manifest = CON.execute(
    f"""
    SELECT
        project_key,
        split_cutoff,
        count(*) FILTER (WHERE split = 'train') AS train_issues,
        count(*) FILTER (WHERE split = 'test') AS test_issues,
        count(*) FILTER (WHERE split = 'train' AND assignee_key IS NOT NULL AND resolutiondate IS NOT NULL) AS train_labeled,
        count(*) FILTER (WHERE split = 'test' AND assignee_key IS NOT NULL AND resolutiondate IS NOT NULL) AS test_labeled,
        count(DISTINCT assignee_key) FILTER (WHERE split = 'train') AS train_distinct_resolvers,
        count(DISTINCT assignee_key) FILTER (WHERE split = 'test') AS test_distinct_resolvers
    FROM read_parquet('{OUT_ISSUES}')
    GROUP BY project_key, split_cutoff
    ORDER BY project_key
    """
).fetchdf()

manifest.to_csv(OUT_MANIFEST, index=False)
print(manifest.to_string(index=False))
