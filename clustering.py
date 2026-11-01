"""Stage 3: K-means on normalised daily curves, with a k/seed sweep, a holdout year and a normalisation check."""
import time
from collections import defaultdict
from itertools import combinations
from math import comb
from pathlib import Path
from statistics import mean

from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import Normalizer, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.sql import Window, functions as F

from common import HOUR_COLS, HOURS, read_table, save_json, write_table

KEY = ['user_id', 'date']
FEATURE_COLS = KEY + ['day_class', 'low']
DAY_COLS = KEY + ['year', 'dow', 'is_weekend', 'day_class', 'low', 'daily_kwh']
MODEL_NAME = 'kmeans_daily_shape'
# (upper bound, name) bands used to name a centre curve.
DAYPARTS = [(6, 'night'), (11, 'morning'), (17, 'daytime'), (24, 'evening')]
BASE_LEVELS = [(0.3, 'low'), (0.5, 'medium'), (float('inf'), 'high')]


def feature_stages(norm):
    # share: p=1 turns kWh into each hour's fraction of the day; peak: p=inf scales the daily maximum to 1.
    p = 1.0 if norm == 'share' else float('inf')
    return [VectorAssembler(inputCols=HOUR_COLS, outputCol='hourly_kwh'),
            Normalizer(inputCol='hourly_kwh', outputCol='features', p=p)]


def kmeans(k, seed, prediction='cluster'):
    return KMeans(k=k, seed=seed, featuresCol='features', predictionCol=prediction, maxIter=200)


def silhouette(df, prediction='cluster'):
    return ClusteringEvaluator(featuresCol='features', predictionCol=prediction,
                               metricName='silhouette', distanceMeasure='squaredEuclidean').evaluate(df)


def contingency(df, col_a, col_b):
    """{(label_a, label_b): days} for two cluster assignments of the same days."""
    rows = df.groupBy(col_a, col_b).count().collect()
    return {(a, b): n for a, b, n in rows}


def adjusted_rand(cells):
    """ARI from contingency counts: 1 = identical partitions, about 0 = chance agreement."""
    n = sum(cells.values())
    rows, cols = defaultdict(int), defaultdict(int)
    for (a, b), count in cells.items():
        rows[a] += count
        cols[b] += count
    index = sum(comb(count, 2) for count in cells.values())
    row_sum = sum(comb(count, 2) for count in rows.values())
    col_sum = sum(comb(count, 2) for count in cols.values())
    expected = row_sum * col_sum / comb(n, 2)
    maximum = (row_sum + col_sum) / 2
    if maximum == expected:
        return 1.0
    return (index - expected) / (maximum - expected)


def best_match_jaccard(cells):
    """For each cluster of the first assignment: Jaccard overlap with its best match in the second."""
    sizes_a, sizes_b = defaultdict(int), defaultdict(int)
    for (a, b), n in cells.items():
        sizes_a[a] += n
        sizes_b[b] += n
    best = defaultdict(float)
    for (a, b), n in cells.items():
        overlap = n / (sizes_a[a] + sizes_b[b] - n)
        best[a] = max(best[a], overlap)
    return dict(best)


def select_days(spark, args):
    daily = read_table(spark, args.data_root, 'daily_curves').where('valid')
    train = daily.where(F.col('year') == args.year)
    counts = train.where(~F.col('user_id').isin(args.exclude_meters)).groupBy('user_id').count()
    meters = counts.where(F.col('count') >= args.min_valid_days).select('user_id')
    train = train.join(meters, 'user_id')
    holdout = None
    if args.holdout_year:
        holdout = daily.where(F.col('year') == args.holdout_year).join(meters, 'user_id')
    return train, holdout, meters.count()


def prepare_features(spark, days, norm):
    """Cached feature vectors. Every fit reads this table, because k-means initialisation depends on row order."""
    transformed = Pipeline(stages=feature_stages(norm)).fit(days).transform(days)
    selected = transformed.select(*FEATURE_COLS, 'features')
    return selected.repartition(spark.sparkContext.defaultParallelism).cache()


def fit_seeds(features, k, seeds):
    """One K-means per seed. Predictions land in columns c<seed> of a single cached DataFrame."""
    preds, runs = features, []
    for seed in seeds:
        start = time.perf_counter()
        model = kmeans(k, seed, f'c{seed}').fit(features)
        seconds = round(time.perf_counter() - start, 2)
        runs.append({'k': k, 'seed': seed, 'seconds': seconds, 'model': model,
                     'cost': model.summary.trainingCost, 'iterations': model.summary.numIter})
        preds = model.transform(preds)
    preds = preds.cache()
    for run in runs:
        run['silhouette'] = silhouette(preds, f"c{run['seed']}")
    return preds, runs


def best_run(runs):
    """Restart with the lowest within-cluster cost."""
    return min(runs, key=lambda run: run['cost'])


def best_cost_hits(runs):
    """Number of seeds within 0.1% of the lowest cost."""
    lowest = best_run(runs)['cost']
    return sum(run['cost'] <= lowest * 1.001 for run in runs)


def seed_aris(preds, seeds):
    """ARI for every pair of seeds."""
    return [adjusted_rand(contingency(preds, f'c{a}', f'c{b}')) for a, b in combinations(seeds, 2)]


def sweep(features, ks, seeds):
    runs, stability = [], []
    for k in ks:
        preds, k_runs = fit_seeds(features, k, seeds)
        aris = seed_aris(preds, seeds)
        preds.unpersist()
        runs += k_runs
        stability.append({'k': k, 'mean_ari': mean(aris) if aris else None, 'min_ari': min(aris, default=None),
                          'best_cost_hits': best_cost_hits(k_runs), 'seeds': len(seeds)})
        silhouettes = [round(run['silhouette'], 3) for run in k_runs]
        print(f"  k={k}: silhouette {silhouettes}, mean ARI {stability[-1]['mean_ari']:.3f}, "
              f"lowest cost reached by {stability[-1]['best_cost_hits']} of {len(seeds)} seeds")
    return runs, stability


def without_model(run):
    return {key: value for key, value in run.items() if key != 'model'}


def mean_silhouette(runs, k):
    return mean(run['silhouette'] for run in runs if run['k'] == k)


def choose(runs, args):
    """Final k (fixed by --k, else the best mean silhouette) and the lowest-cost seed at that k."""
    swept = sorted({run['k'] for run in runs})
    means = {candidate: mean_silhouette(runs, candidate) for candidate in swept}
    k = args.k or max(means, key=means.get)
    candidates = [run for run in runs if run['k'] == k]
    if not candidates:  # --k outside the sweep range
        return k, args.seeds[0]
    return k, best_run(candidates)['seed']


def chosen_model(runs, k, seed, features):
    """The sweep's fitted model for (k, seed)."""
    for run in runs:
        if run['k'] == k and run['seed'] == seed:
            return run['model']
    return kmeans(k, seed).fit(features)  # --k outside the sweep range


def as_pipeline(norm, kmeans_model, prediction):
    """Feature stages and a fitted K-means model as one PipelineModel."""
    kmeans_model.setPredictionCol(prediction)
    return PipelineModel(stages=feature_stages(norm) + [kmeans_model])


def band(value, bands):
    """Name of the first (upper bound, name) band whose bound lies above value."""
    for upper, name in bands:
        if value < upper:
            return name
    return bands[-1][1]


def describe(centre):
    """Readable label from the high-load window (hours within 85% of the maximum) and the base load."""
    top = max(centre)
    total = sum(centre)
    high = [h for h in HOURS if centre[h] >= 0.85 * top]
    base_ratio = min(centre) / top
    if len(high) >= 18:
        label = 'flat all day'
    else:
        when = band(sum(high) / len(high), DAYPARTS)
        kind = 'plateau' if len(high) >= 6 else 'peak'
        base = band(base_ratio, BASE_LEVELS)
        label = f'{when} {kind} {high[0]:02d}-{high[-1] + 1:02d}h, {base} base'
    return {'label': label, 'peak_hour': centre.index(top), 'peak_to_mean': top / (total / 24),
            'base_to_peak': base_ratio, 'night_share': sum(centre[0:6]) / total,
            'business_share': sum(centre[8:18]) / total}


def with_distance(spark, preds, centres):
    """Euclidean distance from each day's feature vector to its own cluster centre."""
    centre_df = spark.createDataFrame(list(enumerate(centres)), 'cluster int, centre array<double>')
    squared_diffs = F.zip_with('shape', 'centre', lambda x, y: (x - y) * (x - y))
    squared_distance = F.aggregate(squared_diffs, F.lit(0.0), lambda acc, v: acc + v)
    return (preds.withColumn('shape', vector_to_array('features'))
            .join(F.broadcast(centre_df), 'cluster')
            .withColumn('distance', F.sqrt(squared_distance))
            .drop('centre'))


def representatives(assigned, per_cluster=3):
    """The days closest to each centre."""
    rank = F.row_number().over(Window.partitionBy('cluster').orderBy('distance'))
    rows = assigned.withColumn('rank', rank).where(F.col('rank') <= per_cluster).collect()
    examples = []
    for r in rows:
        examples.append({'cluster': r.cluster, 'user_id': r.user_id, 'date': r.date, 'daily_kwh': r.daily_kwh,
                         'shape': list(r.shape), 'distance': r.distance})
    return examples


def shares(assigned):
    total = assigned.count()
    rows = assigned.groupBy('cluster').count().collect()
    return {r.cluster: r['count'] / total for r in rows}


def normalisation_check(spark, train, assigned, norm, k, seeds):
    """Refit with another normalisation at the same k and compare the assignments."""
    features = prepare_features(spark, train, norm)
    preds, runs = fit_seeds(features, k, seeds)
    preds.unpersist()
    features.unpersist()
    best = best_run(runs)
    alternative = as_pipeline(norm, best['model'], 'alt_cluster')
    both = alternative.transform(assigned.drop('hourly_kwh', 'features'))
    cells = contingency(both, 'cluster', 'alt_cluster')
    return {'norm': norm, 'seed': best['seed'], 'ari': adjusted_rand(cells), 'jaccard': best_match_jaccard(cells)}


def assign(spark, model, days, centres, split):
    labelled = with_distance(spark, model.transform(days), centres)
    return labelled.withColumn('split', F.lit(split)).cache()


def fit_final(spark, args, train, holdout, k, seed, kmeans_model):
    model = as_pipeline(args.norm, kmeans_model, 'cluster')
    model.write().overwrite().save(f"{args.data_root.rstrip('/')}/models/{MODEL_NAME}")
    centres = [c.tolist() for c in kmeans_model.clusterCenters()]
    described = [{'cluster': i, 'centre': c, **describe(c)} for i, c in enumerate(centres)]
    assigned = assign(spark, model, train, centres, 'train')
    result = {'year': args.year, 'holdout_year': args.holdout_year, 'norm': args.norm, 'k': k, 'seed': seed,
              'excluded_meters': args.exclude_meters, 'centres': described,
              'train_silhouette': silhouette(assigned), 'train_shares': shares(assigned),
              'representatives': representatives(assigned)}
    if args.robust_norm not in ('none', args.norm):
        check = normalisation_check(spark, train, assigned, args.robust_norm, k, args.seeds)
        result['normalisation_check'] = check
    output = assigned
    if holdout is not None:
        tested = assign(spark, model, holdout, centres, 'holdout')
        result.update(holdout_days=tested.count(), holdout_silhouette=silhouette(tested),
                      holdout_shares=shares(tested))
        output = output.unionByName(tested)
    columns = DAY_COLS + ['cluster', 'distance', 'split']
    write_table(output.select(*columns).coalesce(4), args.data_root, 'daily_clusters')
    return result


def run(spark, args, log):
    train, holdout, n_meters = select_days(spark, args)
    features = prepare_features(spark, train, args.norm)
    n_days = features.count()
    excluded = ', '.join(args.exclude_meters) or 'none'
    print(f'  training days: {n_days} from {n_meters} meters (year {args.year}, norm={args.norm}, '
          f'excluded meters: {excluded})')

    ks = list(range(args.k_min, args.k_max + 1))
    with log.stage('cluster_sweep', ks=ks, seeds=args.seeds, training_days=n_days) as info:
        runs, stability = sweep(features, ks, args.seeds)
        sweep_metrics = [without_model(run) for run in runs]
        info.update(runs=sweep_metrics, stability=stability)

    k, seed = choose(runs, args)
    with log.stage('cluster_final', k=k, seed=seed) as info:
        model = chosen_model(runs, k, seed, features)
        result = fit_final(spark, args, train, holdout, k, seed, model)
        result.update(training_days=n_days, meters=n_meters, runs=sweep_metrics, stability=stability)
        info.update(train_silhouette=result['train_silhouette'],
                    holdout_silhouette=result.get('holdout_silhouette'),
                    normalisation_check=result.get('normalisation_check'))
    features.unpersist()
    save_json(result, Path(args.output) / 'clustering.json')

    print(f"  final k={k} seed={seed}: train silhouette {result['train_silhouette']:.3f}, "
          f"holdout silhouette {result.get('holdout_silhouette')}")
    check = result.get('normalisation_check')
    if check:
        print(f"  same k with {check['norm']} normalisation: ARI {check['ari']:.3f}, "
              f"best-match Jaccard {check['jaccard']}")
    for c in result['centres']:
        share = result['train_shares'].get(c['cluster'], 0)
        print(f"  cluster {c['cluster']}: {c['label']} share {share:.2%}")
    return result
