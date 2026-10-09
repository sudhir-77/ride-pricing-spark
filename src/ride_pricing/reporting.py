"""Write metrics, CSV artifacts, plots, and the README results table."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ride_pricing.metrics import json_safe


FORECAST_CSV = "forecast_test_predictions.csv"
TOP_ZONES_CSV = "top50_zones.csv"
SURGE_CSV = "surge_zone_hour.csv"
SWITCHBACK_CSV = "switchback_experiments.csv"
NAIVE_CSV = "naive_ab_cells.csv"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(payload), indent=2) + "\n", encoding="utf-8")


def write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _style_ax(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_wape(path: Path, wape_model: float, wape_naive: float) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    labels = ["LightGBM", "Seasonal naive"]
    values = [wape_model, wape_naive]
    ax.bar(labels, values, color=["#1f4e79", "#9aa5b1"])
    ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)
    _style_ax(ax, "April WAPE, top zones", "", "WAPE")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_forecast_example(path: Path, frame: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    show = frame.sort_values("ts").head(24 * 7)
    ax.plot(show["ts"], show["actual"], label="Actual", color="#222222", linewidth=1.4)
    ax.plot(show["ts"], show["pred_lightgbm"], label="LightGBM", color="#1f4e79", linewidth=1.2)
    ax.plot(show["ts"], show["pred_seasonal_naive"], label="Seasonal naive", color="#c47b2b", linewidth=1.0)
    _style_ax(ax, "Busiest zone, first week of April", "Hour", "Trips")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_multipliers(path: Path, multipliers: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.hist(multipliers, bins=25, color="#1f4e79", edgecolor="white")
    ax.axvline(1.0, color="#c47b2b", linewidth=1.2, label="Flat price")
    _style_ax(ax, "Optimized surge multipliers", "Multiplier", "Zone-hours")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_switchback(path: Path, estimates: np.ndarray, true_effect: float, coverage: float) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.hist(estimates, bins=30, color="#1f4e79", edgecolor="white")
    ax.axvline(true_effect, color="#c47b2b", linewidth=1.4, label="True effect")
    _style_ax(ax, f"Switchback estimates (coverage {coverage:.3f})", "Estimated effect", "Experiments")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _num(value: float) -> str:
    return f"{float(value):.6f}"


def results_markdown(payload: dict) -> str:
    lines = [
        "| Criterion | Value | Threshold | Passed |",
        "|---|---:|---:|:---:|",
    ]
    for item in payload["success_criteria"]:
        high = item["threshold_high"]
        if high is None:
            threshold = _num(item["threshold"])
        else:
            threshold = f"{_num(item['threshold'])} to {_num(high)}"
        passed = "yes" if item["passed"] else "no"
        lines.append(
            f"| {item['name']} | {_num(item['value'])} | {item['op']} {threshold} | {passed} |"
        )
    forecast = payload["metrics"]["forecast"]
    simulator = payload["metrics"]["simulator"]
    switch = payload["metrics"]["switchback"]
    lines.extend(
        [
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| April top-zone WAPE, LightGBM | {_num(forecast['april_top50_wape_lightgbm'])} |",
            f"| April top-zone WAPE, seasonal naive | {_num(forecast['april_top50_wape_seasonal_naive'])} |",
            f"| March validation WAPE, LightGBM | {_num(forecast['val_wape_lightgbm'])} |",
            f"| March validation WAPE, seasonal naive | {_num(forecast['val_wape_seasonal_naive'])} |",
            f"| WAPE improvement vs naive | {_num(forecast['wape_improvement_vs_naive'])} |",
            f"| Simulated trips, optimized | {simulator['completed_trips_optimized']:.4f} |",
            f"| Simulated trips, flat | {simulator['completed_trips_flat']:.4f} |",
            f"| Trip lift | {_num(simulator['trips_lift'])} |",
            f"| Avg rider price, optimized | {_num(simulator['avg_rider_price_optimized'])} |",
            f"| Avg rider price, flat | {_num(simulator['avg_rider_price_flat'])} |",
            f"| Switchback 95% CI coverage | {_num(switch['ci_coverage'])} |",
            f"| Switchback absolute bias | {_num(switch['switchback_bias'])} |",
            f"| Naive user-level absolute bias | {_num(switch['naive_ab_bias'])} |",
            f"| Bias ratio (switchback / naive) | {_num(switch['bias_ratio'])} |",
        ]
    )
    demand = payload["metrics"].get("demand_11m")
    spark = payload["metrics"].get("spark")
    if demand and spark:
        lines.extend(
            [
                f"| April top-zone WAPE, 11-month LightGBM | {_num(demand['april_top50_wape'])} |",
                f"| April top-zone WAPE, 11-month seasonal naive | {_num(demand['april_top50_wape_seasonal_naive'])} |",
                f"| Parent April top-zone WAPE | {_num(demand['parent_april_top50_wape'])} |",
                f"| Spark vs DuckDB match rate | {_num(spark['parity_match_rate'])} |",
                f"| Spark ETL seconds | {spark['runtime_s']:.2f} |",
                f"| Spark total trips | {int(spark['total_trips'])} |",
            ]
        )
    lines.extend(
        [
            f"| Runtime seconds | {payload['runtime_seconds']:.2f} |",
            f"| Peak RSS GiB | {payload['hardware']['peak_rss_gb']:.4f} |",
            "",
            "Values are fractions (0.25 means 25%), except prices in USD, trip totals, seconds, and GiB.",
            "This table is written by `python run.py` from `results/metrics.json`.",
        ]
    )
    return "\n".join(lines)


def update_readme(readme_path: Path, section: str) -> None:
    text = readme_path.read_text(encoding="utf-8")
    start = "<!-- RESULTS_START -->"
    end = "<!-- RESULTS_END -->"
    if start not in text or end not in text:
        raise RuntimeError("README is missing RESULTS markers")
    pre, rest = text.split(start, 1)
    _, post = rest.split(end, 1)
    updated = pre + start + "\n\n" + section.strip() + "\n\n" + end + post
    readme_path.write_text(updated, encoding="utf-8")


def render_resume_notes(payload: dict) -> str:
    forecast = payload["metrics"]["forecast"]
    simulator = payload["metrics"]["simulator"]
    switch = payload["metrics"]["switchback"]
    wape = forecast["april_top50_wape_lightgbm"]
    naive = forecast["april_top50_wape_seasonal_naive"]
    improve = forecast["wape_improvement_vs_naive"]
    lift = simulator["trips_lift"]
    price_opt = simulator["avg_rider_price_optimized"]
    price_flat = simulator["avg_rider_price_flat"]
    ratio = switch["bias_ratio"]
    coverage = switch["ci_coverage"]
    naive_bias = switch["naive_ab_bias"]
    sb_bias = switch["switchback_bias"]

    def r4(value: float) -> str:
        return f"{float(value):.4f}"

    return f"""# Resume notes

## Draft bullets

- Forecast NYC zone-hour taxi demand with LightGBM at {r4(wape)} WAPE on held-out April, versus {r4(naive)} for a seasonal-naive baseline (relative reduction {r4(improve)}).
- Built a marketplace simulator and surge-pricing optimizer that raised simulated completed trips by {r4(lift)} over flat pricing, with average rider price {r4(price_opt)} versus {r4(price_flat)} under the flat multiplier.
- Designed switchback experiments that held absolute interference bias to {r4(sb_bias)} ({r4(ratio)} of the naive user-level A/B bias of {r4(naive_bias)}), with {r4(coverage)} coverage of the true effect by nominal 95% CIs across 500 simulations.

## Skills and tools

Python, SQL, DuckDB, pandas, NumPy, LightGBM, SciPy, statsmodels, matplotlib, pytest. Time-series forecasting, marketplace simulation, constrained optimization, and experiment analysis with clustered standard errors.

## Project summary

This project forecasts yellow-taxi pickups by zone and hour, then uses those forecasts as base demand in a small marketplace simulator. A grid search sets a surge multiplier for each zone-hour, and simulated switchback experiments check that a block-randomized analysis recovers the known pricing effect while a user-level A/B test does not.

## GitHub description

Forecast NYC taxi zone-hour demand, optimize a simulated surge multiplier, and compare switchback experiments with a user-level A/B test.
"""


def render_extension_notes(payload: dict) -> str:
    """Resume notes for the Spark extension. Every digit is formatted from metrics.json."""
    spark = payload["metrics"]["spark"]
    demand = payload["metrics"]["demand_11m"]
    airflow = payload["metrics"]["airflow"]
    forecast = payload["metrics"]["forecast"]

    def r4(value: float) -> str:
        return f"{float(value):.4f}"

    trips = int(spark["total_trips"])
    months = int(spark["n_months"])
    match = r4(spark["parity_match_rate"])
    wape_11 = r4(demand["april_top50_wape"])
    naive_11 = r4(demand["april_top50_wape_seasonal_naive"])
    history = int(demand["n_history_months"])
    n_tasks = int(airflow["n_tasks"])
    parent_wape = r4(forecast["april_top50_wape_lightgbm"])
    return f"""# Resume notes

## Draft bullets

- Built a PySpark ETL over {trips} NYC taxi trips ({months} months) with data-quality checks, matching DuckDB features on {match} of overlapping zone-hour rows.
- Defined the extract, validate, transform, train, and evaluate steps as an Airflow DAG ({n_tasks} tasks, statically validated) and ran the same tasks end to end with a local runner.
- Retrained the demand model on {history} months of history at {wape_11} WAPE on held-out April 2024, versus {naive_11} for seasonal-naive.
- Re-ran the parent DuckDB and LightGBM pipeline at {parent_wape} April top-zone WAPE on the same held-out month.

## Skills and tools

Python, SQL, PySpark, DuckDB, Airflow DAG definitions, LightGBM, MLflow, pandas, NumPy, pytest. Local-mode Spark ETL, feature parity checks, time-series forecasting, and static DAG validation with a local runner.

## Project summary

This project extends a zone-hour taxi demand model with a local PySpark cleaning and aggregation job over 12 months of TLC yellow trips. An Airflow DAG defines extract, validate, transform, train, and evaluate; a local runner executes those tasks, and the retrained model is scored on held-out April 2024.

## GitHub description

PySpark ETL for a year of NYC taxi trips, checked against DuckDB, with an Airflow DAG and an 11-month demand model.
"""
