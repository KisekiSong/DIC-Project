"""Stage 1: raw UCI text file -> validated long-format readings (Delta)."""
from pyspark.sql import functions as F

from common import write_table


def load_wide(spark, raw_path):
    # Semicolon-separated with decimal commas; one row per timestamp, one column per meter.
    df = spark.read.option('sep', ';').option('header', True).option('inferSchema', False).csv(raw_path)
    return df.withColumnRenamed(df.columns[0], 'ts_label')


def to_long(wide):
    meters = [c for c in wide.columns if c != 'ts_label']
    long = wide.unpivot(['ts_label'], meters, 'user_id', 'kw_raw')
    ts = F.to_timestamp('ts_label', 'yyyy-MM-dd HH:mm:ss')
    return (long
            .withColumn('ts', ts)
            .withColumn('kw', F.regexp_replace('kw_raw', ',', '.').cast('double'))
            # A label marks the end of its 15-minute interval.
            .withColumn('interval_start', F.col('ts') - F.expr('INTERVAL 15 MINUTES'))
            .withColumn('year', F.year('interval_start')))


def count_where(condition, name):
    return F.sum(condition.cast('int')).alias(name)


def validation_report(wide, long):
    ts = F.col('ts')
    stamps = wide.select(F.to_timestamp('ts_label', 'yyyy-MM-dd HH:mm:ss').alias('ts'))
    s = stamps.agg(F.count('*').alias('rows'), F.countDistinct('ts').alias('distinct_ts'),
                   count_where(ts.isNull(), 'invalid_ts'),
                   F.min('ts').alias('first_ts'), F.max('ts').alias('last_ts')).first()
    expected = int((s.last_ts - s.first_ts).total_seconds() // 900) + 1  # one label every 15 minutes

    kw, raw = F.col('kw'), F.col('kw_raw')
    readings = long.agg(F.count('*').alias('readings'),
                        count_where(kw.isNull() & raw.isNotNull(), 'unparseable'),
                        count_where(raw.isNull(), 'missing'),
                        count_where(kw < 0, 'negative'),
                        count_where(kw == 0, 'zero')).first()
    return {'timestamp_rows': s.rows, 'distinct_timestamps': s.distinct_ts,
            'duplicate_timestamps': s.rows - s.distinct_ts, 'invalid_timestamps': s.invalid_ts,
            'expected_timestamps': expected, 'gap_timestamps': expected - s.distinct_ts,
            'first_ts': s.first_ts, 'last_ts': s.last_ts,
            'meters': len(wide.columns) - 1, **readings.asDict()}


def run(spark, args, log):
    with log.stage('ingest') as info:
        wide = load_wide(spark, args.raw).cache()
        long = to_long(wide)
        report = validation_report(wide, long)
        valid = long.where(F.col('ts').isNotNull() & F.col('kw').isNotNull() & (F.col('kw') >= 0))
        if report['duplicate_timestamps']:
            # Deduplication needs a shuffle, so it only runs when duplicates exist.
            valid = valid.dropDuplicates(['user_id', 'ts'])
        valid = valid.select('user_id', 'ts', 'interval_start', 'kw', 'year')
        write_table(valid, args.data_root, 'readings', partition_by=['year'])
        wide.unpersist()
        info['report'] = report
    for key, value in report.items():
        print(f'  {key}: {value}')
    return report
