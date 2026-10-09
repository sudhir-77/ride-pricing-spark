"""Compare Spark and DuckDB zone-hour features on the parent overlap window."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ride_pricing.data import aggregate_zone_hour
from ride_pricing.year_config import OVERLAP_END, OVERLAP_MONTHS, OVERLAP_START, PARITY_MAX_ABS_DIFF
from spark.grid import naive_timestamps

FEATURE_COLUMNS = ["trips", "hour", "dow", "month", "is_weekend"]


def spark_overlap_slice(panel: pd.DataFrame) -> pd.DataFrame:
    """Hours in the parent window, restricted to zones that have a trip there.

    The year panel also contains zones that only appear in 2023. Those zones are
    all zeros in January-April, and the DuckDB panel does not include them.
    """
    start = pd.Timestamp(OVERLAP_START)
    end = pd.Timestamp(OVERLAP_END)
    frame = panel.copy()
    frame["ts"] = naive_timestamps(frame["ts"]).astype("datetime64[ns]")
    part = frame.loc[(frame["ts"] >= start) & (frame["ts"] < end)].copy()
    totals = part.groupby("zone_id", sort=False)["trips"].sum()
    keep = totals[totals > 0].index
    part = part.loc[part["zone_id"].isin(keep)].copy()
    return part.sort_values(["zone_id", "ts"]).reset_index(drop=True)


def duckdb_overlap(paths: list[Path], temp_dir: Path | None = None) -> pd.DataFrame:
    return aggregate_zone_hour(
        paths,
        start=pd.Timestamp(OVERLAP_START),
        end=pd.Timestamp(OVERLAP_END),
        temp_dir=temp_dir,
    )


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in ["zone_id", "ts", *FEATURE_COLUMNS] if column not in frame.columns]
    if missing:
        raise KeyError(f"feature frame is missing {missing}")
    out = frame.loc[:, ["zone_id", "ts", *FEATURE_COLUMNS]].copy()
    out["zone_id"] = out["zone_id"].astype(int)
    out["ts"] = naive_timestamps(out["ts"]).astype("datetime64[ns]")
    for column in FEATURE_COLUMNS:
        out[column] = out[column].astype(float)
    out = out.sort_values(["zone_id", "ts"]).reset_index(drop=True)
    if out.duplicated(["zone_id", "ts"]).any():
        raise ValueError("feature frame has duplicate zone-hour keys")
    return out


def compare_feature_frames(spark_df: pd.DataFrame, duck_df: pd.DataFrame) -> dict:
    """Match rate is 1 only when every key agrees and every numeric gap is <= 1e-9."""
    left = _prepare(spark_df)
    right = _prepare(duck_df)
    merged = left.merge(right, on=["zone_id", "ts"], how="outer", suffixes=("_spark", "_duck"))
    both = merged.dropna(subset=[f"{FEATURE_COLUMNS[0]}_spark", f"{FEATURE_COLUMNS[0]}_duck"])
    matched = np.ones(len(both), dtype=bool)
    max_diff = 0.0
    for column in FEATURE_COLUMNS:
        gap = np.abs(
            both[f"{column}_spark"].to_numpy(dtype=float) - both[f"{column}_duck"].to_numpy(dtype=float)
        )
        if len(gap):
            if not np.isfinite(gap).all():
                raise ValueError(f"non-finite {column} while comparing features")
            max_diff = max(max_diff, float(gap.max()))
        matched &= gap <= PARITY_MAX_ABS_DIFF
    rows_matched = int(matched.sum()) if len(both) else 0
    rows_spark = int(len(left))
    rows_duckdb = int(len(right))
    if rows_spark == 0 or rows_duckdb == 0 or rows_spark != rows_duckdb:
        match_rate = 0.0
    else:
        match_rate = float(rows_matched) / float(rows_duckdb)
    return {
        "months_compared": list(OVERLAP_MONTHS),
        "rows_spark": rows_spark,
        "rows_duckdb": rows_duckdb,
        "rows_matched": rows_matched,
        "max_abs_diff": float(max_diff),
        "match_rate": match_rate,
    }
