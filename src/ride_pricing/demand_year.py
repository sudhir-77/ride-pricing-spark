"""11-month demand model. April 2024 is held out. Early stopping uses March only."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from ride_pricing.config import (
    LGB_EARLY_STOPPING_ROUNDS,
    LGB_EARLY_STOPPING_ROUNDS_QUICK,
    LGB_N_ESTIMATORS,
    LGB_N_ESTIMATORS_QUICK,
    SEED,
    TEST_START,
    TOP_N_ZONES,
    TRAIN_START,
    VAL_START,
)
from ride_pricing.features import add_lag_features, supervised_frame
from ride_pricing.forecast import predict_nonnegative, seasonal_naive, train_lightgbm
from ride_pricing.metrics import wape


def assign_split_11m(ts: pd.Series) -> pd.Series:
    """Train is every supervised hour before 18 March 2024. Test is April."""
    stamps = pd.Series(pd.to_datetime(ts))
    labels = pd.Series("train", index=stamps.index, dtype="object")
    labels.loc[stamps >= pd.Timestamp(VAL_START)] = "val"
    labels.loc[stamps >= pd.Timestamp(TEST_START)] = "test"
    return labels


def fit_demand(panel: pd.DataFrame, quick: bool, tracking_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Fit on all zones. Rank the scored zones on the parent training window only."""
    featured = supervised_frame(add_lag_features(panel))
    featured["split"] = assign_split_11m(featured["ts"])
    if featured.duplicated(["zone_id", "ts"]).any():
        raise RuntimeError("duplicate zone-hour rows in the 11-month frame")
    train = featured.loc[featured["split"] == "train"].copy()
    val = featured.loc[featured["split"] == "val"].copy()
    test = featured.loc[featured["split"] == "test"].copy()
    if train.empty or val.empty or test.empty:
        raise RuntimeError("11-month split is missing train, val, or test rows")
    if train["ts"].max() >= pd.Timestamp(VAL_START) or val["ts"].max() >= pd.Timestamp(TEST_START):
        raise RuntimeError("11-month split boundaries overlap")
    if test["ts"].min() < pd.Timestamp(TEST_START) or test["ts"].max() >= pd.Timestamp("2024-05-01"):
        raise RuntimeError("11-month test split is not April 2024")
    rank = train.loc[
        (train["ts"] >= pd.Timestamp(TRAIN_START)) & (train["ts"] < pd.Timestamp(VAL_START))
    ]
    if rank.empty:
        raise RuntimeError("parent training window has no rows for zone ranking")
    zone_totals = rank.groupby("zone_id", sort=False)["trips"].sum().sort_values(ascending=False)
    n_top = min(TOP_N_ZONES, int(zone_totals.shape[0]))
    top_ids = [int(zone) for zone in zone_totals.head(n_top).index.tolist()]
    n_estimators = LGB_N_ESTIMATORS_QUICK if quick else LGB_N_ESTIMATORS
    stopping = LGB_EARLY_STOPPING_ROUNDS_QUICK if quick else LGB_EARLY_STOPPING_ROUNDS
    model = train_lightgbm(train, val, n_estimators=n_estimators, early_stopping_rounds=stopping)
    best_iteration = int(getattr(model, "best_iteration", 0) or n_estimators)
    val_pred = predict_nonnegative(model, val)
    test_pred = predict_nonnegative(model, test)
    val_naive = seasonal_naive(val["lag_168"].to_numpy())
    test_naive = seasonal_naive(test["lag_168"].to_numpy())
    val_top = val["zone_id"].astype(int).isin(top_ids).to_numpy()
    test_top_mask = test["zone_id"].astype(int).isin(top_ids).to_numpy()
    predictions = pd.DataFrame(
        {
            "zone": test.loc[test_top_mask, "zone_id"].astype(int).to_numpy(),
            "hour_ts": pd.to_datetime(test.loc[test_top_mask, "ts"]).map(
                lambda stamp: pd.Timestamp(stamp).strftime("%Y-%m-%dT%H:%M:%S")
            ),
            "actual": test.loc[test_top_mask, "trips"].to_numpy(dtype=float),
            "pred": test_pred[test_top_mask],
            "pred_seasonal_naive": test_naive[test_top_mask],
        }
    )
    predictions.insert(
        0,
        "row_id",
        [f"{int(zone)}|{ts}" for zone, ts in zip(predictions["zone"], predictions["hour_ts"])],
    )
    predictions = predictions.sort_values(["zone", "hour_ts"]).reset_index(drop=True)
    if not predictions["row_id"].is_unique:
        raise RuntimeError("April prediction ids are not unique")
    run_id = _log_mlflow(
        tracking_dir,
        quick=quick,
        n_estimators=n_estimators,
        best_iteration=best_iteration,
        train_rows=int(len(train)),
        val_rows=int(len(val)),
        val_wape=wape(val.loc[val_top, "trips"].to_numpy(), val_pred[val_top]),
    )
    summary = {
        "best_iteration": best_iteration,
        "n_top_zones": n_top,
        "top_zones": top_ids,
        "train_rows": int(len(train)),
        "val_rows": int(len(val)),
        "test_rows": int(len(test)),
        "test_top_rows": int(len(predictions)),
        "val_wape_lightgbm": wape(val.loc[val_top, "trips"].to_numpy(), val_pred[val_top]),
        "val_wape_seasonal_naive": wape(val.loc[val_top, "trips"].to_numpy(), val_naive[val_top]),
        "mlflow_run_id": run_id,
        "seed": SEED,
        "n_history_months": 11,
    }
    return predictions, summary


def _log_mlflow(
    tracking_dir: Path,
    *,
    quick: bool,
    n_estimators: int,
    best_iteration: int,
    train_rows: int,
    val_rows: int,
    val_wape: float,
) -> str:
    import os

    import mlflow

    os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    mlflow.set_experiment("nyc_taxi_demand_11m")
    with mlflow.start_run(run_name="quick" if quick else "full") as run:
        mlflow.log_params(
            {
                "quick": str(bool(quick)),
                "n_estimators": str(int(n_estimators)),
                "best_iteration": str(int(best_iteration)),
                "train_rows": str(int(train_rows)),
                "val_rows": str(int(val_rows)),
                "seed": str(int(SEED)),
            }
        )
        mlflow.log_metric("val_wape_top_zones", float(val_wape))
        return run.info.run_id


def log_test_wape(tracking_dir: Path, run_id: str, test_wape: float) -> None:
    """Record the held-out WAPE after the model is already fit. Not a tuning input."""
    import os

    import mlflow

    os.environ.setdefault("MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING", "false")
    tracking_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.resolve().as_uri())
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("test_wape_top_zones", float(test_wape))


def plot_demand_wape(path: Path, wape_11m: float, wape_parent: float, wape_naive: float) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    labels = ["11-month LightGBM", "Parent LightGBM", "Seasonal naive"]
    values = [wape_11m, wape_parent, wape_naive]
    ax.bar(labels, values, color=["#1f4e79", "#5b8fad", "#9aa5b1"])
    ax.set_ylim(0, max(values) * 1.25 if max(values) > 0 else 1)
    ax.set_title("April top-zone WAPE")
    ax.set_ylabel("WAPE")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def write_train_outputs(cache: Path, predictions: pd.DataFrame, summary: dict) -> None:
    folder = cache / "features"
    folder.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(folder / "april_predictions.parquet", index=False)
    (folder / "train_summary.json").write_text(json.dumps(summary) + "\n", encoding="utf-8")


def load_train_outputs(cache: Path) -> tuple[pd.DataFrame, dict]:
    folder = cache / "features"
    predictions = pd.read_parquet(folder / "april_predictions.parquet")
    summary = json.loads((folder / "train_summary.json").read_text(encoding="utf-8"))
    return predictions, summary
