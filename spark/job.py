"""spark-submit entry for the local ETL: extract, validate, and transform.

Example (from the project folder, after the virtualenv is installed):

    spark-submit --master local[2] --driver-memory 1g spark/job.py --cache-dir data --output-dir results

Training and evaluation stay in `python run.py`. This process does not start a
cluster.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("PYSPARK_SUBMIT_ARGS", "--driver-memory 1g pyspark-shell")
    os.environ.setdefault("SPARK_DRIVER_MEMORY", "1g")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    src = root / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    parser = argparse.ArgumentParser(description="Local PySpark zone-hour ETL")
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args(argv)
    cache = args.cache_dir
    out = args.output_dir
    cache.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    config_path = cache / "run_config.json"
    config_path.write_text(
        json.dumps(
            {
                "quick": bool(args.quick),
                "output_dir": str(out.resolve()),
                "cache_dir": str(cache.resolve()),
                "project_root": str(root),
                "device": "cpu",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    os.environ["RIDE_PRICING_RUN_CONFIG"] = str(config_path)
    from ride_pricing.tasks import extract, transform, validate

    extract()
    validate()
    transform()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
