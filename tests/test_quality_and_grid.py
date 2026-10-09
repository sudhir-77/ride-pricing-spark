import pandas as pd

from spark.quality import QualityThresholds, assess_month


def _stats(**overrides) -> dict:
    base = {
        "month": "2024-01",
        "raw": 100,
        "null_pickup": 0,
        "null_zone": 0,
        "null_fare": 0,
        "null_distance": 0,
        "fare_out": 4,
        "distance_out": 10,
    }
    base.update(overrides)
    return base


def test_quality_rates_match_hand_counts():
    thresholds = QualityThresholds(
        max_null_rate=0.01,
        max_fare_out_of_range_rate=0.05,
        max_distance_out_of_range_rate=0.20,
        min_rows=50,
    )
    # 4/100 = 0.04 fare failures, under 0.05. 10/100 distance failures, equal to 0.20.
    assert assess_month(_stats(), thresholds) == []
    # 6/100 = 0.06 fare failures, over 0.05.
    fare_failures = assess_month(_stats(fare_out=6), thresholds)
    assert len(fare_failures) == 1
    assert "fare" in fare_failures[0]
    # 2/100 null pickups, over 0.01.
    null_failures = assess_month(_stats(null_pickup=2), thresholds)
    assert any("tpep_pickup_datetime" in item for item in null_failures)


def test_quality_rejects_a_short_month_and_a_bad_pin():
    thresholds = QualityThresholds(min_rows=1000, exact_rows={"2024-01": 100})
    failures = assess_month(_stats(raw=100), thresholds)
    assert any("minimum" in item for item in failures)
    pinned = QualityThresholds(min_rows=1, exact_rows={"2024-01": 90})
    assert any("pinned" in item for item in assess_month(_stats(), pinned))


def test_calendar_matches_duckdb_extract():
    import duckdb

    from spark.grid import calendar_frame

    hours = pd.date_range("2024-01-01", periods=24 * 10, freq="h").union(
        pd.date_range("2024-03-09", periods=72, freq="h")
    )
    frame = pd.DataFrame({"ts": hours})
    con = duckdb.connect(database=":memory:")
    try:
        con.register("hours", frame)
        duck = con.execute(
            """
            SELECT
                ts,
                EXTRACT(hour FROM ts)::INTEGER AS hour,
                EXTRACT(dow FROM ts)::INTEGER AS dow,
                EXTRACT(month FROM ts)::INTEGER AS month,
                CASE WHEN EXTRACT(dow FROM ts) IN (0, 6) THEN 1 ELSE 0 END AS is_weekend
            FROM hours
            ORDER BY ts
            """
        ).fetchdf()
    finally:
        con.close()
    calendar = calendar_frame(duck["ts"])
    assert calendar["hour"].tolist() == duck["hour"].astype(int).tolist()
    assert calendar["dow"].tolist() == duck["dow"].astype(int).tolist()
    assert calendar["month"].tolist() == duck["month"].astype(int).tolist()
    assert calendar["is_weekend"].tolist() == duck["is_weekend"].astype(int).tolist()
    # 2024-01-01 was a Monday. DuckDB dow is 1. Sunday 2024-01-07 is 0.
    monday = duck.loc[pd.to_datetime(duck["ts"]) == pd.Timestamp("2024-01-01")]
    sunday = duck.loc[pd.to_datetime(duck["ts"]) == pd.Timestamp("2024-01-07")]
    assert int(monday["dow"].iloc[0]) == 1
    assert int(sunday["dow"].iloc[0]) == 0
    assert int(sunday["is_weekend"].iloc[0]) == 1


def test_grid_zero_fills_without_changing_the_trip_total():
    from datetime import datetime

    from spark.grid import complete_zone_hour_grid

    counts = pd.DataFrame(
        {
            "zone_id": [1, 1],
            "ts": pd.to_datetime(["2024-01-01 00:00:00", "2024-01-01 02:00:00"]),
            "trips": [2, 1],
        }
    )
    panel = complete_zone_hour_grid(counts, datetime(2024, 1, 1), datetime(2024, 1, 1, 3))
    assert len(panel) == 3
    assert panel["trips"].tolist() == [2, 0, 1]
    assert int(panel["trips"].sum()) == 3
    assert panel["hour"].tolist() == [0, 1, 2]
