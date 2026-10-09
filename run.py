"""Reproduce the project: python run.py or python run.py --quick."""

from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "4")
os.environ.setdefault("SPARK_DRIVER_MEMORY", "1g")
os.environ.setdefault("PYSPARK_SUBMIT_ARGS", "--driver-memory 1g pyspark-shell")
os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parent
    src = root / "src"
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from ride_pricing.extension import run_project

    parser = argparse.ArgumentParser(description="Ride-hail forecasting plus a local Spark ETL and Airflow DAG")
    parser.add_argument("--quick", action="store_true", help="Synthetic subset, under a few minutes")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="cpu is the supported device")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for metrics, CSV, and plots")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Directory for downloaded or synthetic data")
    args = parser.parse_args(argv)
    run_project(
        quick=args.quick,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        device=args.device,
        update_docs=args.output_dir is None and not args.quick,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
