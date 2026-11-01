"""Stage 2: readings -> one row per (user, day) with a 24-value hourly kWh curve and day labels (Delta)."""
from functools import reduce
from operator import add

from pyspark.sql import functions as F

from common import (HOUR_COLS, HOURS, READINGS_PER_DAY, dst_transition_days, pt_public_holidays, read_table,
                    write_table)


def hour_kwh(h):
    """kWh used in hour h: four 15-minute average-kW readings, each lasting a quarter of an hour."""
    in_hour = F.hour('interval_start') == h
    return (F.sum(F.when(in_hour, F.col('kw'))) / 4).alias(HOUR_COLS[h])


def build_daily(readings):
    # 24 conditional sums in one aggregation: the whole curve needs a single shuffle.
    hourly = [hour_kwh(h) for h in HOURS]
    daily = (readings
             .groupBy('user_id', F.to_date('interval_start').alias('date'))
             .agg(F.count('kw').alias('n_readings'),
                  F.sum((F.col('kw') == 0).cast('int')).alias('zero_readings'),
                  *hourly))
    hour_values = [F.coalesce(F.col(c), F.lit(0.0)) for c in HOUR_COLS]
    return (daily
            .withColumn('year', F.year('date'))
            .withColumn('dow', F.dayofweek('date'))  # 1 = Sunday ... 7 = Saturday
            .withColumn('is_weekend', F.col('dow').isin(1, 7))
            .withColumn('daily_kwh', reduce(add, hour_values)))


def label_validity(daily, years):
    dst = [d.isoformat() for d in dst_transition_days(years)]
    reason = (F.when(F.col('n_readings') < READINGS_PER_DAY, 'incomplete')
              .when(F.col('date').cast('string').isin(dst), 'dst_transition')
              .when(F.col('daily_kwh') <= 0, 'zero_consumption')
              # likely an outage or a meter start
              .when(F.col('zero_readings') >= READINGS_PER_DAY // 2, 'mostly_zero'))
    return daily.withColumn('exclusion', reason).withColumn('valid', reason.isNull())


def label_day_types(daily, years):
    """day_class (weekday / sat / sun / holiday) and low (under half the user's median valid day that year)."""
    holidays = []
    for year in years:
        holidays += [d.isoformat() for d in pt_public_holidays(year)]
    is_holiday = F.col('date').cast('string').isin(holidays) & ~F.col('is_weekend')
    day_class = (F.when(is_holiday, 'holiday')
                 .when(F.col('dow') == 7, 'sat')
                 .when(F.col('dow') == 1, 'sun')
                 .otherwise('weekday'))
    median_kwh = F.expr('percentile_approx(daily_kwh, 0.5)').alias('median_kwh')
    typical = daily.where('valid').groupBy('user_id', 'year').agg(median_kwh)
    return (daily.withColumn('day_class', day_class)
            .join(typical, ['user_id', 'year'], 'left')
            .withColumn('low', F.col('daily_kwh') < 0.5 * F.col('median_kwh')))


def coverage(daily):
    """Per year: valid days and how many meters are usable for most of the year."""
    per_user = (daily.groupBy('year', 'user_id')
                .agg(F.sum(F.col('valid').cast('int')).alias('valid_days'), F.count('*').alias('days')))
    mostly_valid = (F.col('valid_days') >= 0.9 * F.col('days')).cast('int')
    return (per_user.groupBy('year')
            .agg(F.countDistinct('user_id').alias('meters'),
                 F.sum('valid_days').alias('valid_days'),
                 F.sum(mostly_valid).alias('meters_90pct_valid'))
            .orderBy('year'))


def run(spark, args, log):
    readings = read_table(spark, args.data_root, 'readings')
    years = sorted(r.year for r in readings.select('year').distinct().collect())
    with log.stage('features', years=years) as info:
        # cached: label_day_types reads it twice
        daily = label_validity(build_daily(readings), years).cache()
        write_table(label_day_types(daily, years), args.data_root, 'daily_curves', partition_by=['year'])
        daily.unpersist()
        stored = read_table(spark, args.data_root, 'daily_curves')
        outcome_rows = stored.groupBy('exclusion').count().collect()
        exclusions = {r['exclusion'] or 'valid': r['count'] for r in outcome_rows}
        cover = [r.asDict() for r in coverage(stored).collect()]
        info.update(exclusions=exclusions, coverage=cover)
    print(f'  day-level outcomes: {exclusions}')
    for row in cover:
        print(f'  coverage {row}')
    return exclusions, cover
