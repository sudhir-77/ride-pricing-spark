"""One local SparkSession with a small heap so the pipeline stays under 3 GB."""

from __future__ import annotations

import os
import shutil

_SPARK = None


def _prepare_env() -> None:
    os.environ.setdefault("SPARK_DRIVER_MEMORY", "1g")
    # The gateway reads this before the JVM starts. Do not export it from a shell
    # that also launches spark-submit; run.py sets it only inside this process.
    os.environ.setdefault("PYSPARK_SUBMIT_ARGS", "--driver-memory 1g pyspark-shell")
    os.environ.setdefault("OMP_NUM_THREADS", "4")


def get_spark():
    """Return the process-wide local session, creating it on first use."""
    global _SPARK
    if _SPARK is not None:
        return _SPARK
    _prepare_env()
    if shutil.which("java") is None and not os.environ.get("JAVA_HOME"):
        raise RuntimeError("Java is required for local PySpark and was not found on PATH")
    from pyspark.sql import SparkSession

    _SPARK = (
        SparkSession.builder.master("local[2]")
        .appName("ride-pricing-spark")
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.default.parallelism", "2")
        .config("spark.sql.adaptive.enabled", "false")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.parquet.inferTimestampNTZ.enabled", "true")
        .config("spark.sql.catalogImplementation", "in-memory")
        .config("spark.driver.host", "127.0.0.1")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.maxResultSize", "512m")
        .config("spark.sql.execution.arrow.pyspark.enabled", "false")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )
    _SPARK.sparkContext.setLogLevel("ERROR")
    return _SPARK


def stop_spark() -> None:
    """Stop the JVM. Safe to call more than once."""
    global _SPARK
    spark = _SPARK
    _SPARK = None
    if spark is None:
        return
    spark.stop()


def maybe_stop_spark() -> None:
    """Release the JVM during a real run. Tests keep the session for later cases."""
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    stop_spark()
