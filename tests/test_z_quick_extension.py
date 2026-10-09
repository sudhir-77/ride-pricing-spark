import json
from pathlib import Path

import pandas as pd

from run import main


def test_quick_entrypoint_runs_parent_and_spark(tmp_path: Path):
    out = tmp_path / "results"
    cache = tmp_path / "data"
    code = main(["--quick", "--output-dir", str(out), "--cache-dir", str(cache)])
    assert code == 0
    payload = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    assert payload["project"] == "ride-pricing-spark"
    assert payload["runtime_seconds"] < 180
    assert payload["hardware"]["peak_rss_gb"] < 3.0
    assert payload["seeds"] == [0]
    spark = payload["metrics"]["spark"]
    assert spark["parity_match_rate"] == 1.0
    assert spark["max_abs_diff"] <= 1e-9
    assert spark["total_trips"] == spark["cleaned_rows"]
    assert payload["metrics"]["airflow"]["all_succeeded"] is True
    assert payload["metrics"]["airflow"]["n_tasks"] == 5
    parity = json.loads((out / "spark_parity.json").read_text(encoding="utf-8"))
    assert parity["rows_spark"] == parity["rows_duckdb"] == parity["rows_matched"]
    assert set(parity) >= {"months_compared", "rows_spark", "rows_duckdb", "rows_matched", "max_abs_diff"}
    stage = json.loads((out / "spark_stage_counts.json").read_text(encoding="utf-8"))
    assert {"raw", "cleaned", "aggregated", "total_trips", "runtime_s"} <= set(stage)
    predictions = pd.read_csv(out / "demand_test_predictions_11m.csv")
    assert {"row_id", "zone", "hour_ts", "actual", "pred", "pred_seasonal_naive"} <= set(predictions.columns)
    stamps = pd.to_datetime(predictions["hour_ts"])
    assert stamps.min() >= pd.Timestamp("2024-04-01")
    assert stamps.max() < pd.Timestamp("2024-05-01")
    log = (out / "dag_run.log").read_text(encoding="utf-8")
    for name in ("extract", "validate", "transform", "train", "evaluate"):
        assert f"task={name} status=success" in log
    assert (out / "demand_wape_11m.png").exists()
    assert (cache / "mlruns").exists()
    # April rows from the 11-month model are not in the training split.
    assert payload["splits"]["demand_11m"]["train"] > 0
    assert payload["splits"]["demand_11m"]["test"] > 0
    manifest = json.loads((cache / "splits" / "split_manifest.json").read_text(encoding="utf-8"))
    parent_forecast = pd.read_csv(out / "forecast_test_predictions.csv")
    assert pd.to_datetime(parent_forecast["ts"]).min() >= pd.Timestamp(manifest["test_start"])
