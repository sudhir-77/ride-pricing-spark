"""Airflow task callables. The DAG file names these; the local runner calls them.

Importing this module does not start Spark. Each task reads RIDE_PRICING_RUN_CONFIG.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

from ride_pricing.data import download_file, tlc_url, verify_file
from ride_pricing.metrics import wape
from ride_pricing.reporting import write_frame, write_json
from ride_pricing.year_config import (
    EARLY_END,
    EARLY_MONTHS,
    OVERLAP_END,
    OVERLAP_MONTHS,
    OVERLAP_START,
    SPARK_MONTHS,
    YEAR_END,
    YEAR_START,
    all_pins,
    pins_for,
)
from spark.quality import QualityThresholds


def load_config() -> dict:
    raw = os.environ.get("RIDE_PRICING_RUN_CONFIG")
    if not raw:
        raise RuntimeError("RIDE_PRICING_RUN_CONFIG is not set")
    return json.loads(Path(raw).read_text(encoding="utf-8"))


def _cache(cfg: dict) -> Path:
    return Path(cfg["cache_dir"])


def _output(cfg: dict) -> Path:
    return Path(cfg["output_dir"])


def _month_path(cache: Path, month: str) -> Path:
    return cache / "raw" / f"yellow_tripdata_{month}.parquet"


def _quick_path(cache: Path) -> Path:
    return cache / "quick" / "synthetic_year.parquet"


def _profile_files(cfg: dict) -> list[tuple[str, Path]]:
    cache = _cache(cfg)
    if cfg["quick"]:
        return [("synthetic", _quick_path(cache))]
    return [(month, _month_path(cache, month)) for month in SPARK_MONTHS]


def _aggregate_jobs(cfg: dict) -> list[tuple[str, Path, object, object]]:
    cache = _cache(cfg)
    if cfg["quick"]:
        return [("synthetic", _quick_path(cache), YEAR_START, YEAR_END)]
    jobs: list[tuple[str, Path, object, object]] = []
    for month in EARLY_MONTHS:
        jobs.append((month, _month_path(cache, month), YEAR_START, EARLY_END))
    for month in OVERLAP_MONTHS:
        jobs.append((month, _month_path(cache, month), OVERLAP_START, OVERLAP_END))
    return jobs


def _thresholds(quick: bool) -> QualityThresholds:
    if quick:
        return QualityThresholds(min_rows=1, exact_rows=None)
    return QualityThresholds(min_rows=1_000_000, exact_rows=all_pins())


def _quality_path(cache: Path) -> Path:
    return cache / "quality" / "quality_report.json"


def _panel_path(cache: Path) -> Path:
    return cache / "features" / "zone_hour.parquet"


def extract() -> None:
    """Download the TLC months, or build the synthetic year in quick mode."""
    cfg = load_config()
    cache = _cache(cfg)
    if cfg["quick"]:
        from spark.synthetic import write_quick_year

        path = write_quick_year(_quick_path(cache))
        manifest = {"quick": True, "files": [str(path)]}
        print(f"quick extract wrote {path.name}", flush=True)
    else:
        files: list[str] = []
        downloads = 0
        for month in SPARK_MONTHS:
            dest = _month_path(cache, month)
            if not dest.exists():
                print(f"downloading {tlc_url(month)}", flush=True)
                download_file(tlc_url(month), dest, pause_seconds=3.0 if downloads else 0.0)
                downloads += 1
            else:
                print(f"using cached {dest.name}", flush=True)
            digest, rows = pins_for(month)
            verify_file(dest, digest, rows)
            files.append(str(dest))
        manifest = {"quick": False, "files": files}
    path = cache / "extract_manifest.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def validate() -> None:
    """Check raw row counts, null rates, and fare and distance ranges."""
    from spark.aggregate import profile_file
    from spark.quality import assess_all

    cfg = load_config()
    cache = _cache(cfg)
    months = []
    spark_runtime = 0.0
    for label, path in _profile_files(cfg):
        if not path.exists():
            raise FileNotFoundError(f"extract did not produce {path}")
        print(f"validating {label}", flush=True)
        stats = profile_file(path, label)
        spark_runtime += float(stats["runtime_s"])
        months.append(stats)
    report = assess_all(months, _thresholds(bool(cfg["quick"])))
    report["spark_runtime_s"] = round(spark_runtime, 3)
    report["raw"] = int(sum(int(item["raw"]) for item in months))
    report["raw_by_month"] = {item["month"]: int(item["raw"]) for item in months}
    write_json(_quality_path(cache), report)
    if not report["passed"]:
        raise RuntimeError("data-quality checks failed: " + "; ".join(report["failures"]))
    print(f"quality checks passed raw_rows={report['raw']}", flush=True)


def transform() -> None:
    """Clean and aggregate to a zone-hour panel. Stops Spark before training."""
    from spark.aggregate import aggregate_panel
    from spark.session import maybe_stop_spark

    cfg = load_config()
    cache = _cache(cfg)
    out = _output(cfg)
    report = json.loads(_quality_path(cache).read_text(encoding="utf-8"))
    print("aggregating zone-hours in Spark", flush=True)
    panel, stage = aggregate_panel(_aggregate_jobs(cfg), YEAR_START, YEAR_END)
    raw_by_month = report["raw_by_month"]
    stage_out = {
        "raw": int(report["raw"]),
        "cleaned": int(stage["cleaned"]),
        "aggregated": int(stage["aggregated"]),
        "total_trips": int(stage["total_trips"]),
        "runtime_s": round(float(report["spark_runtime_s"]) + float(stage["aggregate_runtime_s"]), 3),
        "months": {
            label: {
                "raw": int(raw_by_month[label]),
                "cleaned": int(stage["cleaned_by_month"].get(label, 0)),
            }
            for label in raw_by_month
        },
    }
    if int(stage_out["total_trips"]) != int(stage_out["cleaned"]):
        raise RuntimeError("cleaned trip count and panel trip total disagree")
    panel_path = _panel_path(cache)
    panel_path.parent.mkdir(parents=True, exist_ok=True)
    panel.to_parquet(panel_path, index=False)
    write_json(out / "spark_stage_counts.json", stage_out)
    print(
        f"spark panel rows={stage_out['aggregated']} trips={stage_out['total_trips']} "
        f"runtime_s={stage_out['runtime_s']}",
        flush=True,
    )
    del panel
    maybe_stop_spark()


def train() -> None:
    """Retrain LightGBM on every supervised hour before 18 March 2024."""
    from ride_pricing.demand_year import fit_demand, write_train_outputs

    cfg = load_config()
    cache = _cache(cfg)
    panel = pd.read_parquet(_panel_path(cache))
    print(f"training 11-month demand model rows={len(panel)}", flush=True)
    predictions, summary = fit_demand(panel, quick=bool(cfg["quick"]), tracking_dir=cache / "mlruns")
    write_train_outputs(cache, predictions, summary)
    print(
        f"train rows={summary['train_rows']} val rows={summary['val_rows']} "
        f"best_iteration={summary['best_iteration']}",
        flush=True,
    )


def evaluate() -> None:
    """Score April, compare Spark features to DuckDB, and write the artifacts."""
    from ride_pricing.demand_year import load_train_outputs, log_test_wape, plot_demand_wape
    from ride_pricing.parity import compare_feature_frames, duckdb_overlap, spark_overlap_slice

    cfg = load_config()
    cache = _cache(cfg)
    out = _output(cfg)
    out.mkdir(parents=True, exist_ok=True)
    predictions, summary = load_train_outputs(cache)
    csv_path = out / "demand_test_predictions_11m.csv"
    write_frame(csv_path, predictions)
    disk = pd.read_csv(csv_path)
    stamps = pd.to_datetime(disk["hour_ts"])
    if stamps.min() < pd.Timestamp("2024-04-01") or stamps.max() >= pd.Timestamp("2024-05-01"):
        raise RuntimeError("11-month predictions are not restricted to April 2024")
    model_wape = wape(disk["actual"].to_numpy(), disk["pred"].to_numpy())
    naive_wape = wape(disk["actual"].to_numpy(), disk["pred_seasonal_naive"].to_numpy())
    log_test_wape(cache / "mlruns", summary["mlflow_run_id"], model_wape)
    parent_wape = _parent_wape(out / "metrics.json")
    panel = pd.read_parquet(_panel_path(cache))
    if cfg["quick"]:
        overlap_paths = [_quick_path(cache)]
    else:
        overlap_paths = [_month_path(cache, month) for month in OVERLAP_MONTHS]
    print("comparing Spark and DuckDB features on the overlap", flush=True)
    parity = compare_feature_frames(
        spark_overlap_slice(panel),
        duckdb_overlap(overlap_paths, temp_dir=cache / "tmp"),
    )
    write_json(
        out / "spark_parity.json",
        {
            "months_compared": parity["months_compared"],
            "rows_spark": parity["rows_spark"],
            "rows_duckdb": parity["rows_duckdb"],
            "rows_matched": parity["rows_matched"],
            "max_abs_diff": parity["max_abs_diff"],
        },
    )
    plot_parent = float(parent_wape) if parent_wape is not None else float(naive_wape)
    plot_demand_wape(out / "demand_wape_11m.png", model_wape, plot_parent, naive_wape)
    summary_out = {
        "april_top50_wape": model_wape,
        "april_top50_wape_seasonal_naive": naive_wape,
        "parent_april_top50_wape": parent_wape,
        "val_wape_lightgbm": summary["val_wape_lightgbm"],
        "val_wape_seasonal_naive": summary["val_wape_seasonal_naive"],
        "best_iteration": summary["best_iteration"],
        "n_top_zones": summary["n_top_zones"],
        "n_history_months": summary["n_history_months"],
        "mlflow_run_id": summary["mlflow_run_id"],
        "train_rows": summary["train_rows"],
        "val_rows": summary["val_rows"],
        "test_rows": summary["test_rows"],
        "parity_match_rate": parity["match_rate"],
        "rows_spark": parity["rows_spark"],
        "rows_duckdb": parity["rows_duckdb"],
        "rows_matched": parity["rows_matched"],
        "max_abs_diff": parity["max_abs_diff"],
    }
    write_json(cache / "features" / "eval_summary.json", summary_out)
    print(
        f"11-month April WAPE={model_wape:.6f} naive={naive_wape:.6f} "
        f"parity={parity['match_rate']:.6f}",
        flush=True,
    )


def _parent_wape(path: Path) -> float | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    forecast = payload.get("metrics", {}).get("forecast", {})
    value = forecast.get("april_top50_wape_lightgbm")
    if value is None:
        return None
    return float(value)
