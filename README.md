# Discovering Daily Electricity Use Patterns with Apache Spark

ID2221 final project. The pipeline turns the UCI ElectricityLoadDiagrams20112014 meter data into one
normalised 24-hour load curve per user-day, groups those curves with K-means, and summarises how often each
user follows each pattern on weekdays and weekends. The two-page report is `report/report.pdf`.

## Pipeline

```text
LD2011_2014.txt (wide: 140,256 timestamps x 370 meters, 15-min kW)
  │ ingest.py     parse ; and decimal commas, unpivot to long, validate, shift labels to interval start
  ▼
delta/readings          51.9M rows  (user_id, ts, interval_start, kw)       partitioned by year
  │ features.py   one aggregation: 24 conditional sums per (user, day) -> hourly kWh curve, validity rules,
  │               day class (weekday / sat / sun / holiday) and low-day flag
  ▼
delta/daily_curves      ~540k rows  (user_id, date, dow, h00..h23, daily_kwh, valid, exclusion, day_class, low)
  │ clustering.py Spark ML: VectorAssembler -> Normalizer -> KMeans; k x seed sweep, lowest-cost restart,
  │               holdout year, refit with the other normalisation as a robustness check
  ▼
delta/daily_clusters    (user_id, date, cluster, distance to centre, split = train | holdout)
models/kmeans_daily_shape   persisted PipelineModel
  │ profiles.py   per-user pattern shares, weekly-structure metrics (weekly.py), Plotly report
  ▼
delta/user_profiles + output/report.html, output/clustering.json, output/run_log.jsonl
(all Delta tables and the model can live on HDFS: see "Run on HDFS")
```

| Stage | Main Spark idea | Output |
|---|---|---|
| `ingest` | CSV source, `unpivot` (wide to long), Delta write partitioned by `year` | validated readings + validation report |
| `features` | single `groupBy` with conditional aggregation (manual pivot, one shuffle) | daily curve table, exclusion counts, coverage per year |
| `cluster` | Spark ML `Pipeline`, `KMeans`, `ClusteringEvaluator`, broadcast join for centre distances | sweep metrics, final model, assignments |
| `profile` | window functions for per-user shares; small count table collected for `weekly.py` | user profiles, HTML report |
| `explore` (opt-in) | same features and K-means over a norm x k x seed grid, scored on weekly structure | `output/explore*/explore.html` |

`weekly.py` holds the weekly-structure metrics (day-type lift, weekly MI, holiday check). They run in plain Python on
a collected table of (user, day class, low, cluster) counts, so the report and the experiment share one definition.

## Data decisions

- **Timestamp convention.** Each label marks the end of a 15-minute interval, so `2011-01-02 00:00:00` belongs to
  1 January. Every reading is shifted back 15 minutes before it is assigned to a day and hour.
- **Time zone.** Labels are Portuguese wall-clock time without an offset. The Spark session and the Python process
  both run in UTC so the labels are never shifted.
- **Units.** Values are average kW over 15 minutes, so the hourly kWh is the sum of four readings divided by 4.
- **Excluded days** (written to the `exclusion` column, never silently dropped):
  `incomplete` (< 96 readings), `dst_transition` (last Sunday of March and October: the source zero-fills or
  double-counts 01:00–02:00), `zero_consumption` (meter not yet active), `mostly_zero` (≥ 48 zero readings,
  likely an outage or a meter start).
- **Day labels.** `day_class` is weekday, sat, sun or holiday (Portuguese national holidays on a weekday, computed
  from Easter; Corpus Christi, 5 Oct, 1 Nov and 1 Dec were suspended in 2013–2015). `low` marks days under half of
  the user's median valid day in that year, a candidate closed or reduced-operation day.
- **Study period.** 2013 is used for fitting (330 meters have ≥ 90% valid days) and 2014 as a holdout year.
  Meters need ≥ 300 valid days in 2013 to be clustered. MT_093 is excluded by default (see Exploration), which
  leaves 330 meters and 119,677 days.
- **Normalisation.** `--norm share` divides each curve by its daily total (L1 `Normalizer`), so clusters describe
  *when* energy is used. `daily_kwh` is kept for interpretation. `--norm peak` divides by the daily maximum instead.

## Run

Requires Python 3.11+ and Java 17. The first Spark launch downloads the Delta Lake JARs through Ivy.

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
mkdir -p data/raw && curl -fL -o data/raw/uci.zip https://archive.ics.uci.edu/static/public/321/electricityloaddiagrams20112014.zip
unzip -o data/raw/uci.zip -d data/raw
.venv/bin/python pipeline.py                    # all stages: share, k = 5, MT_093 excluded, peak check
.venv/bin/python pipeline.py cluster profile    # re-run later stages only
.venv/bin/python pipeline.py cluster profile --norm peak --robust-norm share   # swap the two normalisations
.venv/bin/python pipeline.py cluster profile --k 0 --exclude-meters ''         # automatic k, every meter
```

Run with `--help` to see every option. Each stage appends its time, settings and key numbers to
`output/run_log.jsonl`.

## Run on HDFS

All table and model paths go through Spark's Hadoop file system layer, so the same code runs on `hdfs://` URIs.
`hdfs/docker-compose.yml` starts a single-node HDFS: the official `apache/hadoop:3.3.6` image, with the
NameNode and DataNode in one container and the data in a Docker volume. The image is amd64-only; Docker Desktop
emulates it on Apple Silicon.

```bash
docker compose -f hdfs/docker-compose.yml up -d
docker exec dic-hdfs hdfs dfs -mkdir -p /elec/raw
docker exec -i dic-hdfs hdfs dfs -put -f - /elec/raw/LD2011_2014.txt < data/raw/LD2011_2014.txt
HADOOP_CONF_DIR=$PWD/hdfs/client-conf .venv/bin/python pipeline.py \
    --raw hdfs://localhost:9000/elec/raw/LD2011_2014.txt --data-root hdfs://localhost:9000/elec --output output/hdfs
docker compose -f hdfs/docker-compose.yml down   # keeps the HDFS volume; add -v to delete it
```

The DataNode registers as `localhost`, and `hdfs/client-conf/hdfs-site.xml` tells the client to connect by that
name (through the published port 9866) and to write a single replica. The NameNode web UI is at
http://localhost:9870.

Verified run (2026-10-22): the raw file is stored as 6 blocks of up to 128 MB (678 MB in total). The Delta tables
take 198 MB (`readings` 117 MB partitioned by year, `daily_curves` 76 MB, `daily_clusters` 4 MB), and the
PipelineModel is saved under `/elec/models`. The run produced the same cluster centres, silhouettes and pattern
shares as the local run. Stage times compared with local disk:

| stage | local (s) | HDFS (s) |
|---|---|---|
| ingest | 25.5 | 29.8 |
| features | 7.2 | 11.1 |
| k sweep (features cached in memory) | 32.8 | 33.1 |
| final model and checks | 8.9 | 12.8 |

## Report

`report/report.md` is the source of the two-page report and `report/report.pdf` the rendered version. Figure 1 is
drawn from `output/clustering.json`, so build the report after running the pipeline:

```bash
.venv/bin/python report/build_report.py
```

The build fails if the PDF exceeds two pages or the source contains a placeholder.

## Results (default configuration)

Share normalisation, k = 5, MT_093 excluded; local[*] on 14 cores, about 80 s end to end (ingest 26 s,
features 7 s, k sweep 33 s, final model and checks 9 s, profiles 1 s).

- Validation: 51,894,720 readings, no duplicate, invalid or missing timestamps, no unparseable or negative values.
- Training data: 119,677 valid days from 330 meters in 2013.
- Five patterns. In the table, lift is P(day type | pattern) / P(day type); a low day is under half of the user's
  median day.

  | pattern | days | Sun lift | weekday-holiday lift | low-day lift |
  |---|---|---|---|---|
  | daytime plateau 09–22h, medium base | 50.3% | 0.91 | 0.77 | 0.11 |
  | evening peak 19–23h, medium base | 18.4% | 0.90 | 0.96 | 0.56 |
  | flat all day (closed day) | 13.8% | 1.56 | 2.24 | 5.88 |
  | daytime plateau 11–22h, low base | 10.2% | 1.00 | 0.86 | 0.18 |
  | office hours 09–19h, medium base | 7.3% | 0.80 | 0.57 | 0.21 |

- Restarts: the lowest-cost solution is reached by 3 of the 5 default seeds, and by 8 of 10 seeds in the
  experiment below.
- Over time: silhouette 0.439 in 2013 and 0.444 on the 2014 holdout. 95.5% of user x weekday/weekend pairs
  keep the same dominant pattern from 2013 to 2014.
- Weekly structure: I(cluster; weekend | user) is 0.071 of 0.864 possible bits, and 79 of 330 users change their
  weekday/weekend pattern mix by more than 0.2. Of the weekday holidays of the 55 users whose Sunday pattern
  differs from their weekday one, 50% take the Sunday pattern, against 15% of their ordinary weekdays.
- Peak normalisation at the same k (`--robust-norm peak`): ARI 0.67 with the share model. Best-match Jaccard is
  0.94 for the low-base plateau, 0.78 for the broad plateau, 0.76 for the evening peak and 0.69 for the closed
  day. The closed-day value is near its ceiling, because peak's closed-day cluster is smaller (10% against 14%).
  The office-hours pattern reaches only 0.46: peak normalisation merges it into a wider 09–20h plateau.

## Exploration: normalisation and k

```bash
.venv/bin/python pipeline.py explore --seeds 1,2,3,4,5,6,7,8,9,10                       # MT_093 excluded
.venv/bin/python pipeline.py explore --seeds 1,2,3,4,5,6,7,8,9,10 --exclude-meters ''   # every meter
```

Silhouette cannot answer "does the clustering separate weekends?", and it is not comparable across the two
feature spaces. The experiment therefore fits each configuration with 10 seeds, keeps the lowest-cost restart and
scores it on:

- **Restart reproducibility**: how many seeds reach the lowest cost (within 0.1%). The final model is the
  lowest-cost restart, so this says how reliably a rerun reproduces it. Mean pairwise ARI is also reported, but it
  mixes in the local optima that the restarts exist to discard.
- **Weekly MI**: I(cluster; weekend | user) in bits. Conditioning on the user means a clustering that only groups
  users scores 0. The upper bound is H(weekend) = 0.864 bits.
- **Holiday → Sunday**: for users whose dominant Sunday pattern differs from their weekday one, the share of
  their weekday holidays that take the Sunday pattern, against the same rate on ordinary weekdays.
- **Per-cluster lift** on Mon–Fri, Sat, Sun, holiday and low days, **per-cluster seed stability**
  (best-match Jaccard against the other seeds), and the **largest meter** in each cluster.

Results for 2013 with MT_093 excluded (330 meters, 119,677 days):

| norm, k | seeds at lowest cost | ARI mean | weekly MI | closed-day cluster: days, lift Sun / holiday / low, Jaccard |
|---|---|---|---|---|
| share, 3 | 9 / 10 | 0.917 | 0.043 | none: low days fall into the evening/high-base cluster (low lift 2.8) |
| share, 4 | 6 / 10 | 0.818 | 0.059 | 14.3%, 1.52 / 2.21 / 5.70, 0.73 |
| **share, 5** | **8 / 10** | **0.913** | **0.071** | **13.8%, 1.56 / 2.24 / 5.88, 0.91** |
| share, 6 | 5 / 10 | 0.779 | 0.071 | 14.8%, plus a 0.6% night-peak cluster that is 25% one meter |
| peak, 4 | 10 / 10 | 1.000 | 0.059 | 11.5%, 1.33 / 2.28 / 4.39, 1.00 |
| peak, 5 | 7 / 10 | 0.819 | 0.057 | 10.3%, 1.39 / 2.42 / 4.81, 0.95 |
| peak, 6 | 7 / 10 | 0.772 | 0.066 | 8.9%, 1.30 / 2.46 / 4.85, 0.96 |

With every meter included, share at k = 5 and k = 6 reaches its lowest cost with only 1 of 10 seeds, and that
solution spends a cluster (0.1% of days) on MT_093 alone.

Findings:

1. **There is a closed-day pattern, and no separate weekend pattern.** From k = 4 on, both normalisations
   produce a "flat all day" curve, which is base load only. It is 4.4–5.9 times over-represented on low days and
   2.2–2.5 times on weekday holidays. No cluster is specific to weekends: the highest Sunday lift is 1.56.
2. **Weekly structure is a small part of the variation.** Weekly MI stays at or below 0.071 bits (under 9% of
   0.864), and user identity explains 63–81% of cluster entropy. Most meters run the same profile all week, and
   weekend differences show up as some users switching to the closed-day pattern. With share at k = 5 there is
   also an office-hours plateau (09–19h) that is under-represented on Sundays (0.80) and holidays (0.57).
3. **Holidays behave like Sundays.** At k = 4–5, 43–51% of the switchers' weekday holidays take their Sunday
   pattern, against 11–16% of their ordinary weekdays. This is an independent check that the closed-day cluster
   reflects operation and not the calendar alone.
4. **MT_093 is excluded on purpose.** It runs a 13:00–17:00 load block all year, which takes 27–59% of its
   daily energy (monthly averages). With it, the lowest-cost share solution at k = 5 gives it a cluster of its own
   (99% of that cluster's days) and only 1 of 10 seeds finds that solution. Without it, 8 of 10 do.
5. **Why share, k = 5.** It is the smallest share model that contains the closed-day pattern reproducibly, it
   adds the office-hours pattern, and it has the highest weekly MI among the reproducible configurations. Peak at
   k = 4 is the most stable configuration and finds the same closed-day pattern, which is why peak is the
   built-in robustness check.
6. Silhouette prefers k = 3–4 because it rewards coarse user types, and k = 3 cannot show closed days. Silhouette
   alone is the wrong criterion for this research question.

### Method notes

- **Restart selection.** Among seeds the pipeline keeps the lowest within-cluster cost, the K-means objective.
  Choosing by silhouette instead picked a higher-cost local optimum (cost 141.8 against 138.6) with a 0.6%
  night-peak cluster.
- **Convergence.** `maxIter` is 200. At 50, several k = 5–6 runs stopped before converging, and that looked like
  seed instability.
- **Determinism.** Spark's k-means initialisation depends on row order as well as on the seed. Every fit starts
  from the same cached, repartitioned feature table, and the final model is the sweep's own fitted run, so a seed
  always means the same model.

## Known limitations and next steps

- Some calendars switch pattern by season, so a month or season breakdown belongs in the profiles.
- Holidays use the national calendar only; municipal holidays are not covered.
- Record stage times under different `--master local[n]` and `--shuffle-partitions` settings for the report.
- HDFS was tested as a single node in Docker, not on a multi-node cluster.
