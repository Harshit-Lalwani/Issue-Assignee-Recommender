---
title: Issue Assignee Recommender
emoji: 🎯
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# Issue-Assignee Recommender

Two-stage recommender (BM25/embedding retrieval &rarr; LightGBM LambdaRank) that ranks the
developers most likely to resolve a newly-filed Jira issue, trained and evaluated on 129K real
issues from 5 open-source projects (Jira's own tracker, Spark, Hadoop, Kafka, Cassandra).

Full source, methodology, and measured results:
[github.com/Harshit-Lalwani/Issue-Assignee-Recommender](https://github.com/Harshit-Lalwani/Issue-Assignee-Recommender)

Try it via the form on this page, or directly:

```bash
curl -X POST https://megaknight9x-issue-assignee-recommender.hf.space/recommend/KAFKA \
  -H "Content-Type: application/json" \
  -d '{"summary": "Consumer group rebalance causes duplicate message processing", "description": "..."}'
```
