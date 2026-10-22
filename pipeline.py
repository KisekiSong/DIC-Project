"""Run the daily electricity-pattern pipeline: ingest -> features -> cluster -> profile."""
import argparse
import importlib

from common import RunLog, create_spark

STAGES = ['ingest', 'features', 'cluster', 'profile']
EXTRA = ['explore']  # experiments: run by name only, never part of 'all'
MODULES = {'ingest': 'ingest', 'features': 'features', 'cluster': 'clustering', 'profile': 'profiles',
           'explore': 'explore'}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('stages', nargs='*', default='all', choices=STAGES + EXTRA + ['all'])
    p.add_argument('--raw', default='data/raw/LD2011_2014.txt', help='raw UCI file (local path or hdfs:// URI)')
    p.add_argument('--data-root', default='data', help='root for Delta tables and models (local or hdfs://)')
    p.add_argument('--output', default='output', help='local directory for reports and run log')
    p.add_argument('--year', type=int, default=2013, help='study year used to fit clusters')
    p.add_argument('--holdout-year', type=int, default=2014,
                   help='later year to test patterns over time (0 = off)')
    p.add_argument('--k-min', type=int, default=3)
    p.add_argument('--k-max', type=int, default=8)
    p.add_argument('--seeds', default='1,2,3,4,5', help='random initialisations compared per k')
    # k = 5 is the smallest k with a reproducible closed-day pattern (see README, Exploration).
    p.add_argument('--k', type=int, default=5, help='final cluster count (0 = best mean silhouette)')
    p.add_argument('--norm', choices=['share', 'peak'], default='share',
                   help='share: divide by daily total; peak: divide by daily maximum')
    p.add_argument('--robust-norm', choices=['share', 'peak', 'none'], default='peak',
                   help='refit with this normalisation at the same k and report how well clusters match')
    p.add_argument('--norms', default='share,peak', help='explore: normalisations to compare')
    p.add_argument('--ks', default='3,4,5,6', help='explore: cluster counts to compare')
    p.add_argument('--min-valid-days', type=int, default=300,
                   help='meters need this many valid study-year days to be clustered')
    # MT_093 runs a unique 13:00-17:00 load block; some seeds spend a whole cluster on it alone.
    p.add_argument('--exclude-meters', default='MT_093',
                   help="comma-separated meters left out of fitting ('' keeps every meter)")
    p.add_argument('--master', default='local[*]')
    p.add_argument('--driver-memory', default='8g')
    p.add_argument('--shuffle-partitions', type=int, default=32)
    args = p.parse_args()
    args.seeds = [int(s) for s in args.seeds.split(',')]
    args.norms = args.norms.split(',')
    args.ks = [int(k) for k in args.ks.split(',')]
    args.exclude_meters = [m for m in args.exclude_meters.split(',') if m]
    return args


def main():
    args = parse_args()
    stages = STAGES if 'all' in args.stages else [s for s in STAGES + EXTRA if s in args.stages]
    spark = create_spark(args.master, args.driver_memory, args.shuffle_partitions)
    log = RunLog(args.output, spark, args)
    try:
        for stage in stages:
            importlib.import_module(MODULES[stage]).run(spark, args, log)
    finally:
        spark.stop()


if __name__ == '__main__':
    main()
