"""Data-quality thresholds for raw TLC files.

Fare and distance gates are checks on the raw files. They do not drop rows from
the zone-hour panel: that panel uses the same zone and pickup-time filter as the
DuckDB feature job, so the two engines can match exactly.
"""

from __future__ import annotations

from dataclasses import dataclass


FARE_MAX = 500.0
DISTANCE_MAX = 100.0


@dataclass(frozen=True)
class QualityThresholds:
    max_null_rate: float = 0.01
    max_fare_out_of_range_rate: float = 0.08
    max_distance_out_of_range_rate: float = 0.20
    min_rows: int = 1
    exact_rows: dict[str, int] | None = None


def _rate(count: int, raw: int) -> float:
    if raw <= 0:
        raise ValueError("raw row count must be positive")
    return float(count) / float(raw)


def assess_month(stats: dict, thresholds: QualityThresholds) -> list[str]:
    """Return human-readable failures. An empty list means the month passed.

    `stats` counts rows. Rates are computed here so a test can pass small integers
    and check the arithmetic by hand.
    """
    label = str(stats["month"])
    raw = int(stats["raw"])
    failures: list[str] = []
    if raw <= 0:
        return [f"{label}: raw row count is {raw}"]
    if raw < int(thresholds.min_rows):
        failures.append(f"{label}: raw rows {raw} < minimum {thresholds.min_rows}")
    expected = None if thresholds.exact_rows is None else thresholds.exact_rows.get(label)
    if expected is not None and raw != int(expected):
        failures.append(f"{label}: raw rows {raw} != pinned {expected}")
    null_fields = (
        ("null_pickup", "tpep_pickup_datetime"),
        ("null_zone", "PULocationID"),
        ("null_fare", "fare_amount"),
        ("null_distance", "trip_distance"),
    )
    for key, column in null_fields:
        rate = _rate(int(stats[key]), raw)
        if rate > thresholds.max_null_rate:
            failures.append(f"{label}: null rate of {column} is {rate:.6f}")
    fare_rate = _rate(int(stats["fare_out"]), raw)
    if fare_rate > thresholds.max_fare_out_of_range_rate:
        failures.append(
            f"{label}: fare outside (0, {FARE_MAX:g}] or null at rate {fare_rate:.6f}"
        )
    distance_rate = _rate(int(stats["distance_out"]), raw)
    if distance_rate > thresholds.max_distance_out_of_range_rate:
        failures.append(
            f"{label}: distance outside [0, {DISTANCE_MAX:g}] or null at rate {distance_rate:.6f}"
        )
    return failures


def assess_all(months: list[dict], thresholds: QualityThresholds) -> dict:
    failures: list[str] = []
    reports: list[dict] = []
    for stats in months:
        raw = int(stats["raw"])
        month_failures = assess_month(stats, thresholds)
        failures.extend(month_failures)
        reports.append(
            {
                "month": stats["month"],
                "raw": raw,
                "null_rate_pickup": _rate(int(stats["null_pickup"]), raw) if raw else None,
                "null_rate_zone": _rate(int(stats["null_zone"]), raw) if raw else None,
                "null_rate_fare": _rate(int(stats["null_fare"]), raw) if raw else None,
                "null_rate_distance": _rate(int(stats["null_distance"]), raw) if raw else None,
                "fare_out_of_range_rate": _rate(int(stats["fare_out"]), raw) if raw else None,
                "distance_out_of_range_rate": _rate(int(stats["distance_out"]), raw) if raw else None,
                "passed": not month_failures,
            }
        )
    return {"passed": not failures, "failures": failures, "months": reports}
