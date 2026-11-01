import json
import os
import sys
import time
from contextlib import contextmanager
from datetime import date, timedelta
from pathlib import Path

from delta import configure_spark_with_delta_pip
from pyspark.sql import SparkSession

HOURS = list(range(24))
HOUR_COLS = [f'h{h:02d}' for h in HOURS]
READINGS_PER_DAY = 96


def create_spark(master='local[*]', driver_memory='8g', shuffle_partitions=32):
    os.environ.setdefault('PYSPARK_PYTHON', sys.executable)
    # Collected timestamps use the process time zone; keep it equal to the Spark session.
    os.environ['TZ'] = 'UTC'
    time.tzset()
    builder = (SparkSession.builder.appName('ElectricityDailyPatterns').master(master)
               .config('spark.driver.memory', driver_memory)
               # Source timestamps carry no time zone; UTC leaves them unshifted.
               .config('spark.sql.session.timeZone', 'UTC')
               .config('spark.sql.shuffle.partitions', str(shuffle_partitions))
               .config('spark.sql.extensions', 'io.delta.sql.DeltaSparkSessionExtension')
               .config('spark.sql.catalog.spark_catalog', 'org.apache.spark.sql.delta.catalog.DeltaCatalog'))
    spark = configure_spark_with_delta_pip(builder).getOrCreate()
    spark.sparkContext.setLogLevel('WARN')
    return spark


def table_path(data_root, name):
    return f"{data_root.rstrip('/')}/delta/{name}"


def read_table(spark, data_root, name):
    return spark.read.format('delta').load(table_path(data_root, name))


def write_table(df, data_root, name, partition_by=None):
    writer = df.write.format('delta').mode('overwrite').option('overwriteSchema', 'true')
    if partition_by:
        writer = writer.partitionBy(*partition_by)
    writer.save(table_path(data_root, name))


def dst_transition_days(years):
    """Last Sundays of March and October: the UCI file zero-fills or double-counts 01:00-02:00."""
    days = []
    for year in years:
        for month in (3, 10):
            last_day = date(year, month + 1, 1) - timedelta(days=1)
            days_after_sunday = (last_day.weekday() + 1) % 7  # weekday(): Monday = 0, Sunday = 6
            days.append(last_day - timedelta(days=days_after_sunday))
    return days


def easter(year):
    """Gregorian Easter Sunday."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    g = (b - (b + 8) // 25 + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return date(year, month, day + 1)


def pt_public_holidays(year):
    """Portuguese national holidays (four of them were suspended in 2013-2015)."""
    sunday = easter(year)
    days = [date(year, m, d) for m, d in [(1, 1), (4, 25), (5, 1), (6, 10), (8, 15), (12, 8), (12, 25)]]
    days += [sunday - timedelta(days=2), sunday]
    if not 2013 <= year <= 2015:
        days += [sunday + timedelta(days=60), date(year, 10, 5), date(year, 11, 1), date(year, 12, 1)]
    return sorted(days)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str))


def load_json(path):
    return json.loads(Path(path).read_text())


class RunLog:
    """Appends stage timings and settings to run_log.jsonl."""

    def __init__(self, output_dir, spark, args):
        self.path = Path(output_dir) / 'run_log.jsonl'
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.base = {'spark': spark.version, 'master': spark.sparkContext.master,
                     'default_parallelism': spark.sparkContext.defaultParallelism,
                     'shuffle_partitions': spark.conf.get('spark.sql.shuffle.partitions'),
                     'args': vars(args)}

    @contextmanager
    def stage(self, name, **extra):
        start = time.perf_counter()
        yield extra
        seconds = round(time.perf_counter() - start, 2)
        print(f'[{name}] {seconds}s')
        record = {'stage': name, 'seconds': seconds, 'at': time.strftime('%Y-%m-%dT%H:%M:%S'),
                  **extra, **self.base}
        with self.path.open('a') as f:
            f.write(json.dumps(record, default=str) + '\n')
