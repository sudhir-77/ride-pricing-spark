"""Small TLC-shaped parquet for quick mode. No network."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ride_pricing.config import SEED
from ride_pricing.year_config import YEAR_END, YEAR_START


def write_trip_parquet(frame: pd.DataFrame, dest: Path) -> Path:
    """Write trips as microsecond timestamps. Spark 3.5 cannot read nanosecond parquet."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(dest, index=False, coerce_timestamps="us", allow_truncated_timestamps=True)
    return dest


def write_quick_year(dest: Path, seed: int = SEED) -> Path:
    """One parquet covering May 2023 through April 2024, plus two rows the filters drop."""
    rng = np.random.default_rng(seed)
    hours = pd.date_range(YEAR_START, YEAR_END, freq="h", inclusive="left")
    zones = np.arange(1, 5, dtype=np.int32)
    ts_grid = np.repeat(hours.to_numpy(), len(zones))
    zone_grid = np.tile(zones, len(hours))
    hour = pd.DatetimeIndex(ts_grid).hour.to_numpy()
    dow = pd.DatetimeIndex(ts_grid).dayofweek.to_numpy()
    rate = 2.0 + zone_grid.astype(float)
    rate = rate + np.isin(hour, [8, 9, 17, 18, 19]).astype(float) * 4.0
    rate = rate + (dow >= 5).astype(float) * -0.5
    rate = np.clip(rate, 0.4, None)
    counts = rng.poisson(rate)
    pickup = np.repeat(ts_grid, counts)
    zone = np.repeat(zone_grid, counts)
    frame = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(pickup),
            "PULocationID": zone.astype("int32"),
        }
    )
    extra = pd.DataFrame(
        {
            "tpep_pickup_datetime": pd.to_datetime(["2023-04-30 12:00:00", "2024-01-01 00:30:00"]),
            "PULocationID": np.array([1, 999], dtype=np.int32),
        }
    )
    frame = pd.concat([frame, extra], ignore_index=True)
    frame["fare_amount"] = 8.0 + frame["PULocationID"].clip(upper=263).astype(float)
    frame["trip_distance"] = 1.2 + (frame["PULocationID"].clip(upper=263).astype(float) % 3)
    return write_trip_parquet(frame, dest)
