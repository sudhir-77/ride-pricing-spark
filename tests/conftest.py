import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("SPARK_DRIVER_MEMORY", "1g")
os.environ.setdefault("PYSPARK_SUBMIT_ARGS", "--driver-memory 1g pyspark-shell")
os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")

import matplotlib

matplotlib.use("Agg")


def pytest_sessionfinish(session, exitstatus):
    import sys

    if "spark.session" not in sys.modules:
        return
    from spark.session import stop_spark

    stop_spark()

