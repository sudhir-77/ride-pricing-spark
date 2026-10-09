"""Merge the parent run with the Spark extension and write metrics.json."""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import psutil

from ride_pricing.dag_runner import run_local
from ride_pricing.metrics import assert_finite, make_criterion
from ride_pricing.pipeline import run_pipeline
from ride_pricing.reporting import render_extension_notes, results_markdown, update_readme, write_json
from ride_pricing.year_config import (
    COVERAGE_MIN,
    PARENT_HEADLINES,
    PARENT_REL_TOL,
    PARITY_MAX_ABS_DIFF,
    PROJECT_SLUG,
    SPARK_ETL_MAX_SECONDS,
)
from spark.session import stop_spark

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class TreePeak:
    """Peak RSS of this process plus children (the Spark JVM is a child)."""

    def __init__(self) -> None:
        self._proc = psutil.Process()
        self.peak = self._sample()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="tree-rss", daemon=True)
        self._thread.start()

    def _sample(self) -> int:
        total = 0
        procs = [self._proc]
        try:
            procs.extend(self._proc.children(recursive=True))
        except psutil.Error:
            pass
        for proc in procs:
            try:
                total += int(proc.memory_info().rss)
            except psutil.Error:
                continue
        return total

    def _loop(self) -> None:
        while not self._stop.wait(0.5):
            rss = self._sample()
            if rss > self.peak:
                self.peak = rss

    def stop(self) -> float:
        if not self._stop.is_set():
            self._stop.set()
            self._thread.join(timeout=2.0)
            rss = self._sample()
            if rss > self.peak:
                self.peak = rss
        return self.peak / (1024**3)


def _hardware(peak_rss_gb: float) -> dict:
    cpu = platform.processor() or platform.machine()
    return {
        "cpu": cpu,
        "ram_gb": round(psutil.virtual_memory().total / (1024**3), 2),
        "gpu": None,
        "python": platform.python_version(),
        "os": platform.platform(),
        "peak_rss_gb": round(float(peak_rss_gb), 4),
    }


def headline_relative_error(parent: dict) -> float:
    forecast = parent["metrics"]["forecast"]
    simulator = parent["metrics"]["simulator"]
    switch = parent["metrics"]["switchback"]
    pairs = (
        (forecast["april_top50_wape_lightgbm"], PARENT_HEADLINES["april_top50_wape_lightgbm"]),
        (forecast["april_top50_wape_seasonal_naive"], PARENT_HEADLINES["april_top50_wape_seasonal_naive"]),
        (simulator["trips_lift"], PARENT_HEADLINES["trips_lift"]),
        (switch["ci_coverage"], PARENT_HEADLINES["ci_coverage"]),
    )
    return float(max(abs(float(got) - float(ref)) / abs(float(ref)) for got, ref in pairs))


def measure_coverage(project_root: Path, output_dir: Path) -> tuple[float, bool]:
    cov_path = output_dir / "coverage.json"
    command = [
        sys.executable,
        "-m",
        "pytest",
        "--cov=src",
        "--cov=spark",
        "--cov-report=term-missing",
        f"--cov-report=json:{cov_path}",
        "-q",
    ]
    print("measuring test coverage", flush=True)
    completed = subprocess.run(command, cwd=project_root)
    if not cov_path.exists():
        return 0.0, False
    data = json.loads(cov_path.read_text(encoding="utf-8"))
    totals = data.get("totals", {})
    percent = totals.get("percent_covered")
    if percent is None:
        covered = float(totals.get("covered_lines", 0))
        statements = float(totals.get("num_statements", 0))
        percent = 0.0 if statements == 0 else 100.0 * covered / statements
    return float(percent) / 100.0, completed.returncode == 0


def _write_config(cache: Path, cfg: dict) -> None:
    path = cache / "run_config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
    os.environ["RIDE_PRICING_RUN_CONFIG"] = str(path)


def run_project(
    *,
    quick: bool = False,
    output_dir: Path | None = None,
    cache_dir: Path | None = None,
    device: str = "cpu",
    update_docs: bool = True,
    project_root: Path | None = None,
    measure_tests: bool | None = None,
) -> dict:
    """Re-run the parent pipeline, then the five DAG tasks, then write one metrics file."""
    started = time.perf_counter()
    tracker = TreePeak()
    root = PROJECT_ROOT if project_root is None else project_root
    out = (root / "results") if output_dir is None else Path(output_dir)
    cache = (root / "data") if cache_dir is None else Path(cache_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)
    run_tests = (not quick) if measure_tests is None else measure_tests
    try:
        parent = run_pipeline(
            quick=quick,
            output_dir=out,
            cache_dir=cache,
            device=device,
            update_docs=False,
            project_root=root,
        )
        _write_config(
            cache,
            {
                "quick": quick,
                "output_dir": str(out),
                "cache_dir": str(cache),
                "project_root": str(root),
                "device": device,
            },
        )
        airflow = run_local(out / "dag_run.log", out / "airflow_check.json")
        payload = _merge(parent, out, cache, quick=quick, airflow=airflow)
        if run_tests:
            coverage, tests_ok = measure_coverage(root, out)
            rel = headline_relative_error(parent)
            parent_item = make_criterion("Parent project", rel, "<=", PARENT_REL_TOL)
            parent_item["passed"] = bool(parent_item["passed"] and tests_ok)
            payload["success_criteria"].append(parent_item)
            payload["success_criteria"].append(
                make_criterion("Test coverage of src/ and spark/", coverage, ">=", COVERAGE_MIN)
            )
            payload["metrics"]["parent_tests_passed"] = bool(tests_ok)
            payload["metrics"]["coverage"] = coverage
        peak = max(tracker.stop(), float(parent["hardware"]["peak_rss_gb"]))
        payload["hardware"] = _hardware(peak)
        payload["runtime_seconds"] = round(time.perf_counter() - started, 3)
        assert_finite(payload)
        write_json(out / "metrics.json", payload)
        if update_docs and not quick:
            (root / "RESUME_NOTES.md").write_text(render_extension_notes(payload), encoding="utf-8")
            update_readme(root / "README.md", results_markdown(payload))
        for item in payload["success_criteria"]:
            flag = "PASS" if item["passed"] else "FAIL"
            print(f"{flag} {item['name']}: {item['value']:.6f} ({item['op']} {item['threshold']})", flush=True)
        print(f"runtime_seconds={payload['runtime_seconds']}", flush=True)
        return payload
    finally:
        stop_spark()
        tracker.stop()


def _merge(parent: dict, out: Path, cache: Path, *, quick: bool, airflow: dict) -> dict:
    summary = json.loads((cache / "features" / "eval_summary.json").read_text(encoding="utf-8"))
    stage = json.loads((out / "spark_stage_counts.json").read_text(encoding="utf-8"))
    parity = json.loads((out / "spark_parity.json").read_text(encoding="utf-8"))
    model_wape = float(summary["april_top50_wape"])
    naive_wape = float(summary["april_top50_wape_seasonal_naive"])
    parent_wape = summary["parent_april_top50_wape"]
    if parent_wape is None:
        raise RuntimeError("parent April WAPE is missing; run the parent pipeline first")
    parent_wape = float(parent_wape)
    match_rate = float(summary["parity_match_rate"])
    max_diff = float(parity["max_abs_diff"])
    parity_item = make_criterion(
        "Spark vs. DuckDB feature parity on overlapping months",
        match_rate,
        ">=",
        1.0,
    )
    parity_item["passed"] = bool(
        parity_item["passed"]
        and max_diff <= PARITY_MAX_ABS_DIFF
        and int(parity["rows_spark"]) == int(parity["rows_duckdb"]) == int(parity["rows_matched"])
    )
    wape_item = make_criterion(
        "Demand WAPE on held-out April 2024 with the 11-month model",
        model_wape,
        "<=",
        parent_wape,
    )
    wape_item["passed"] = bool(wape_item["passed"] and model_wape < naive_wape)
    criteria = list(parent["success_criteria"]) + [
        parity_item,
        wape_item,
        make_criterion("Pipeline throughput", float(stage["runtime_s"]), "<=", SPARK_ETL_MAX_SECONDS),
        make_criterion(
            "Airflow DAG",
            1.0
            if (
                airflow["n_tasks"] == 5
                and airflow["acyclic"]
                and airflow["all_succeeded"]
                and airflow["task_order"] == ["extract", "validate", "transform", "train", "evaluate"]
            )
            else 0.0,
            ">=",
            1.0,
        ),
    ]
    payload = {
        "project": PROJECT_SLUG,
        "success_criteria": criteria,
        "metrics": dict(parent["metrics"]),
        "baseline": dict(parent["baseline"]),
        "splits": dict(parent["splits"]),
        "seeds": [0],
        "hardware": dict(parent["hardware"]),
        "runtime_seconds": 0.0,
    }
    payload["metrics"]["spark"] = {
        "total_trips": int(stage["total_trips"]),
        "raw_rows": int(stage["raw"]),
        "cleaned_rows": int(stage["cleaned"]),
        "aggregated_rows": int(stage["aggregated"]),
        "runtime_s": float(stage["runtime_s"]),
        "n_months": 12,
        "parity_match_rate": match_rate,
        "rows_spark": int(parity["rows_spark"]),
        "rows_duckdb": int(parity["rows_duckdb"]),
        "rows_matched": int(parity["rows_matched"]),
        "max_abs_diff": max_diff,
        "months_compared": list(parity["months_compared"]),
    }
    payload["metrics"]["demand_11m"] = {
        "april_top50_wape": model_wape,
        "april_top50_wape_seasonal_naive": naive_wape,
        "parent_april_top50_wape": parent_wape,
        "val_wape_lightgbm": float(summary["val_wape_lightgbm"]),
        "val_wape_seasonal_naive": float(summary["val_wape_seasonal_naive"]),
        "best_iteration": int(summary["best_iteration"]),
        "n_top_zones": int(summary["n_top_zones"]),
        "n_history_months": int(summary["n_history_months"]),
        "mlflow_run_id": summary["mlflow_run_id"],
    }
    payload["metrics"]["airflow"] = {
        "dag_id": airflow["dag_id"],
        "n_tasks": int(airflow["n_tasks"]),
        "acyclic": bool(airflow["acyclic"]),
        "all_succeeded": bool(airflow["all_succeeded"]),
    }
    payload["metrics"]["quick"] = bool(quick)
    payload["splits"]["demand_11m"] = {
        "train": int(summary["train_rows"]),
        "val": int(summary["val_rows"]),
        "test": int(summary["test_rows"]),
    }
    payload["baseline"]["parent_april_top50_wape_lightgbm"] = parent_wape
    payload["baseline"]["demand_11m_april_top50_wape_seasonal_naive"] = naive_wape
    payload["baseline"]["spec_seasonal_naive_wape"] = 0.193
    return payload
