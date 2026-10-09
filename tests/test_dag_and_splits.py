import pandas as pd

from ride_pricing.dag_check import EXPECTED_ORDER, parse_dag_file, parse_dag_source, static_errors
from ride_pricing.dag_runner import run_local
from ride_pricing.demand_year import assign_split_11m
from ride_pricing.parity import compare_feature_frames


def test_real_dag_is_a_five_task_chain():
    spec = parse_dag_file()
    assert static_errors(spec) == []
    assert spec.n_tasks == 5
    assert spec.task_order == EXPECTED_ORDER
    assert spec.acyclic is True
    assert spec.schedule == "0 6 * * *"
    assert spec.dag_id == "nyc_taxi_demand"


def test_a_cycle_fails_static_validation():
    source = """
from airflow import DAG
from airflow.operators.python import PythonOperator

with DAG(dag_id="nyc_taxi_demand", schedule="0 6 * * *") as dag:
    extract_task = PythonOperator(task_id="extract", python_callable=extract)
    validate_task = PythonOperator(task_id="validate", python_callable=validate)
    transform_task = PythonOperator(task_id="transform", python_callable=transform)
    train_task = PythonOperator(task_id="train", python_callable=train)
    evaluate_task = PythonOperator(task_id="evaluate", python_callable=evaluate)
    extract_task >> validate_task >> transform_task >> train_task >> evaluate_task >> extract_task
"""
    spec = parse_dag_source(source)
    assert spec.acyclic is False
    assert static_errors(spec)


def test_local_runner_logs_fake_tasks_in_order(tmp_path):
    seen: list[str] = []

    def _make(name: str):
        def _run() -> None:
            seen.append(name)

        return _run

    callables = {name: _make(name) for name in EXPECTED_ORDER}
    check = run_local(
        tmp_path / "dag_run.log",
        tmp_path / "airflow_check.json",
        callables=callables,
    )
    assert seen == EXPECTED_ORDER
    assert check["all_succeeded"] is True
    assert check["n_tasks"] == 5
    log = (tmp_path / "dag_run.log").read_text(encoding="utf-8")
    assert log.count("status=success") == 5
    assert "task=extract" in log and "task=evaluate" in log


def test_local_runner_records_a_failed_task(tmp_path):
    def _ok() -> None:
        return None

    def _bad() -> None:
        raise RuntimeError("validate failed")

    callables = {name: _ok for name in EXPECTED_ORDER}
    callables["validate"] = _bad
    try:
        run_local(tmp_path / "dag_run.log", tmp_path / "airflow_check.json", callables=callables)
    except RuntimeError as exc:
        assert "validate failed" in str(exc)
    else:
        raise AssertionError("runner should have raised")
    text = (tmp_path / "airflow_check.json").read_text(encoding="utf-8")
    assert '"all_succeeded": false' in text
    assert "status=failed" in (tmp_path / "dag_run.log").read_text(encoding="utf-8")


def test_eleven_month_split_keeps_april_out_of_training():
    stamps = pd.to_datetime(
        [
            "2023-06-01 00:00:00",
            "2024-03-17 23:00:00",
            "2024-03-18 00:00:00",
            "2024-03-31 23:00:00",
            "2024-04-01 00:00:00",
            "2024-04-30 23:00:00",
        ]
    )
    labels = assign_split_11m(stamps)
    assert list(labels) == ["train", "train", "val", "val", "test", "test"]
    assert set(labels.loc[stamps >= pd.Timestamp("2024-04-01")]) == {"test"}
    assert "test" not in set(labels.loc[stamps < pd.Timestamp("2024-03-18")])


def test_feature_compare_is_exact_for_a_hand_built_gap():
    left = pd.DataFrame(
        {
            "zone_id": [1, 1],
            "ts": pd.to_datetime(["2024-04-01 00:00:00", "2024-04-01 01:00:00"]),
            "trips": [3, 4],
            "hour": [0, 1],
            "dow": [1, 1],
            "month": [4, 4],
            "is_weekend": [0, 0],
        }
    )
    right = left.copy()
    same = compare_feature_frames(left, right)
    assert same["rows_spark"] == 2
    assert same["rows_duckdb"] == 2
    assert same["rows_matched"] == 2
    assert same["max_abs_diff"] == 0.0
    assert same["match_rate"] == 1.0
    shifted = left.copy()
    shifted.loc[1, "trips"] = 9
    gap = compare_feature_frames(left, shifted)
    assert gap["rows_matched"] == 1
    assert gap["max_abs_diff"] == 5.0
    assert gap["match_rate"] < 1.0
