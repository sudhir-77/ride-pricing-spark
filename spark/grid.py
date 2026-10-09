"""Turn Spark zone-hour counts into a complete panel with DuckDB-matching calendar fields."""

from __future__ import annotations

import numpy as np
import pandas as pd


def naive_timestamps(ts: pd.Series) -> pd.Series:
    stamps = pd.to_datetime(ts)
    if getattr(stamps.dt, "tz", None) is not None:
        stamps = stamps.dt.tz_localize(None)
    return stamps


def duckdb_dow(dayofweek: pd.Series) -> pd.Series:
    """Map pandas Monday=0..Sunday=6 onto DuckDB EXTRACT(dow) Sunday=0..Saturday=6."""
    return ((dayofweek.astype(int) + 1) % 7).astype("int8")


def calendar_frame(ts: pd.Series) -> pd.DataFrame:
    stamps = naive_timestamps(ts)
    dow = duckdb_dow(stamps.dt.dayofweek)
    return pd.DataFrame(
        {
            "hour": stamps.dt.hour.astype("int8").to_numpy(),
            "dow": dow.to_numpy(),
            "month": stamps.dt.month.astype("int8").to_numpy(),
            "is_weekend": dow.isin((0, 6)).astype("int8").to_numpy(),
        },
        index=stamps.index,
    )


def complete_zone_hour_grid(
    counts: pd.DataFrame,
    start,
    end,
) -> pd.DataFrame:
    """Zero-fill every zone that appears at least once, for every hour in [start, end)."""
    if counts.empty:
        raise ValueError("counts are empty")
    frame = counts.loc[:, ["zone_id", "ts", "trips"]].copy()
    frame["ts"] = naive_timestamps(frame["ts"]).astype("datetime64[ns]")
    frame["zone_id"] = frame["zone_id"].astype("int16")
    frame["trips"] = frame["trips"].astype("int64")
    frame = frame.groupby(["zone_id", "ts"], as_index=False)["trips"].sum()
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    if end_ts <= start_ts:
        raise ValueError("end must be after start")
    hours = pd.date_range(start_ts, end_ts, freq="h", inclusive="left")
    in_window = (frame["ts"] >= start_ts) & (frame["ts"] < end_ts)
    if not bool(in_window.all()):
        raise ValueError("counts contain hours outside the panel window")
    zones = np.sort(frame["zone_id"].unique())
    index = pd.MultiIndex.from_product([zones, hours], names=["zone_id", "ts"])
    panel = frame.set_index(["zone_id", "ts"]).reindex(index, fill_value=0).reset_index()
    if len(panel) != len(zones) * len(hours):
        raise RuntimeError("zone-hour grid has an unexpected number of rows")
    if int(panel["trips"].sum()) != int(frame["trips"].sum()):
        raise RuntimeError("zero-fill changed the trip total")
    calendar = calendar_frame(panel["ts"])
    panel["hour"] = calendar["hour"].to_numpy()
    panel["dow"] = calendar["dow"].to_numpy()
    panel["month"] = calendar["month"].to_numpy()
    panel["is_weekend"] = calendar["is_weekend"].to_numpy()
    panel["zone_id"] = panel["zone_id"].astype("int16")
    panel["trips"] = panel["trips"].astype("int32")
    return panel
