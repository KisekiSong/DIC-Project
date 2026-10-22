# Report evidence record

Working record for `report.md` / `report.pdf`. It is not part of the submitted report.

## 1. Requirement coverage

Sources: the proposal (`Lab/FINAL_PROJECT_PROPOSAL_EN.md`, 26 Sep, the later of the two versions;
`Final_Project_Proposal.pdf`, 22 Sep) and the course introduction (`01_introduction.pdf`, slide 9: the project is
pass/fail and is assessed on source code and report).

| Requirement | Source | Where | Status |
|---|---|---|---|
| Two-page report on dataset, methodology, results, limitations | proposal §4, last sentence | whole report, 2 pages (checked by `build_report.py`) | covered |
| Raw files in HDFS | proposal §2 | §3 Pipeline, Table 2 | covered (single-node HDFS in Docker) |
| Delta Lake for validated readings and daily features | proposal §2 | §3 Pipeline | covered |
| PySpark / Spark SQL preparation and aggregation | proposal §2, §4 | §3 Pipeline | covered |
| Spark ML pipeline: feature transformation, K-means, persistence | proposal §2 | §3 Clustering | covered |
| Plotly visualisation | proposal §2 | §3 Pipeline (one sentence) | covered |
| Checks for duplicates, invalid timestamps, missing readings | proposal §4 | §2 | covered |
| Hourly daily curves; normalisation explored; consumption kept | proposal §4 | §3 Clustering, §4 | covered |
| Several k and random initialisations; silhouette; agreement between repeated assignments; interpretability | proposal §4 | §3 Evaluation, §4 | covered |
| Later observations to test usefulness over time | proposal §4 | §4 (2014 holdout) | covered |
| Weekday/weekend proportions from valid days | proposal §4 | §4 weekly paragraph | covered |
| Processing and clustering times | proposal §4 | Table 2 | covered (one run each) |
| Reproducible code and execution instructions | proposal §4 | repository README | covered outside the report |

## 2. Key numbers and their sources

All runs use the code at the commit that adds this file. Local outputs are in `output/`; the HDFS run is in
`output/hdfs/`.

| Claim in the report | Source |
|---|---|
| 140,256 timestamps, 51,894,720 readings, no duplicates, gaps or invalid values | `output/run_log.jsonl`, stage `ingest` |
| 540,570 user-days; 428,691 valid; 105,096 / 3,823 / 2,960 excluded | stage `features` (`exclusions`) |
| 98% of zero-consumption days before first consumption (103,160 of 105,096) | ad hoc Spark check on `delta/daily_curves`, 2026-10-22 |
| 330 meters with ≥ 90% valid days in 2013 | stage `features` (`coverage`) |
| 711 MB raw file, 6 HDFS blocks | `hdfs fsck` (710,998,915 bytes, 6 blocks, HEALTHY) |
| Table 1 shares and lifts | `daily_clusters` recomputed with `weekly.summarise`; matches `output/explore_excl_MT_093/explore.json` (share, k = 5) |
| Mean silhouette per k (0.461, 0.486, 0.437, ..., 0.239) | `output/clustering.json`, `runs` |
| Seeds at lowest cost and flat-cluster Jaccard (6/10 and 0.73 at k = 4; 8/10 and 0.91 at k = 5) | `output/explore_excl_MT_093/explore.json` |
| All meters: 1/10 seeds, 0.1% cluster with 99% MT_093 | `output/explore/explore.json` (share, k = 5) |
| MT_093: 27–59% of daily energy 13:00–17:00 (monthly means, 2013) | ad hoc Spark check on `delta/daily_curves` |
| Peak normalisation: 10/10 seeds at k = 3 and 4; flat 10.3% and 09–20 h plateau 15.4% at k = 5 | `output/explore_excl_MT_093/explore.json` |
| Peak refit at k = 5: ARI 0.67, Jaccard 0.94 and 0.69 | `output/clustering.json`, `normalisation_check` |
| Weekly MI 0.071 of 0.864 bits; user explains 73%; median shift 0.04; 79 of 330 users; 55 users, 50% vs 15% | `weekly.summarise` on `daily_clusters` (train split) |
| 2014 silhouette 0.444; shares within 1 percentage point; 95.5% of 660 pairs | `output/clustering.json`; profile stage output |
| 50 of 330 users differ between winter and summer | ad hoc Spark check on `daily_clusters` (Dec–Feb vs Jun–Aug) |
| Table 2 times; identical centres on HDFS | `output/run_log.jsonl`, `output/hdfs/run_log.jsonl`; centre comparison of both `clustering.json` files |

## 3. Open items

- The header has no author or group names, like the proposal. The course runs projects in groups of three, so
  add the group members if the submission needs them.
- The timings come from one run each on one laptop; no repeated runs or scaling experiment exist.
- Seasonality is reported as a limitation only; no seasonal profiles were computed.
- The report has not been reviewed by a teacher or another reader.
