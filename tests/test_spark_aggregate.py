from datetime import datetime

import pandas as pd

from ride_pricing.data import aggregate_zone_hour
from ride_pricing.parity import compare_feature_frames
from spark.aggregate import aggregate_panel, profile_file


def _tiny_trips(path):
    rows = [
        ("2024-01-01 00:10:00", 1, 12.5, 1.2),
        ("2024-01-01 00:40:00", 1, 9.0, 0.8),
        ("2024-01-01 01:05:00", 1, 20.0, 3.0),
        ("2024-01-01 03:00:00", 1, 15.0, 2.0),
        ("2024-01-01 02:15:00", 999, 11.0, 1.0),
        ("2024-01-01 02:15:00", 2, 8.0, 1.5),
    ]
    frame = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime([row[0] for row in rows]),
            "PULocationID": [row[1] for row in rows],
            "fare_amount": [row[2] for row in rows],
            "trip_distance": [row[3] for row in rows],
        }
    )
    frame.to_parquet(path, index=False, coerce_timestamps="us", allow_truncated_timestamps=True)


def test_spark_counts_match_duckdb_on_a_known_file(tmp_path):
    path = tmp_path / "tiny.parquet"
    _tiny_trips(path)
    start = datetime(2024, 1, 1, 0, 0, 0)
    end = datetime(2024, 1, 1, 3, 0, 0)
    stats = profile_file(path, "tiny")
    assert stats["raw"] == 6
    assert stats["null_pickup"] == 0
    assert stats["fare_out"] == 0
    panel, stage = aggregate_panel([("tiny", path, start, end)], start, end)
    assert stage["cleaned"] == 4
    assert int(panel["trips"].sum()) == 4
    duck = aggregate_zone_hour([path], start=pd.Timestamp(start), end=pd.Timestamp(end), temp_dir=tmp_path / "tmp")
    compared = compare_feature_frames(panel, duck)
    assert compared["rows_spark"] == compared["rows_duckdb"] == compared["rows_matched"]
    assert compared["max_abs_diff"] <= 1e-9
    assert compared["match_rate"] == 1.0
    zone_one = panel.loc[panel["zone_id"] == 1, "trips"].tolist()
    assert zone_one == [2, 1, 0]
