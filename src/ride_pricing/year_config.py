"""Constants for the 12-month Spark extension. Parent windows stay in config.py."""

from __future__ import annotations

from datetime import datetime

from ride_pricing.config import TLC_ROWS, TLC_SHA256

# May 2023 through April 2024. The last four months are the parent overlap.
SPARK_MONTHS = (
    "2023-05",
    "2023-06",
    "2023-07",
    "2023-08",
    "2023-09",
    "2023-10",
    "2023-11",
    "2023-12",
    "2024-01",
    "2024-02",
    "2024-03",
    "2024-04",
)
EARLY_MONTHS = SPARK_MONTHS[:8]
OVERLAP_MONTHS = SPARK_MONTHS[8:]

YEAR_START = datetime(2023, 5, 1)
YEAR_END = datetime(2024, 5, 1)  # exclusive
EARLY_END = datetime(2024, 1, 1)  # exclusive; overlap files own January onward
OVERLAP_START = datetime(2024, 1, 1)
OVERLAP_END = datetime(2024, 5, 1)

PROJECT_SLUG = "ride-pricing-spark"

# Headline numbers from the verified parent run. The extension re-runs that
# pipeline and requires each of these to land within 2% relative.
PARENT_HEADLINES = {
    "april_top50_wape_lightgbm": 0.14612673860775385,
    "april_top50_wape_seasonal_naive": 0.19296595644565825,
    "trips_lift": 0.21309416610259224,
    "ci_coverage": 0.942,
}
PARENT_REL_TOL = 0.02
SPARK_ETL_MAX_SECONDS = 30 * 60
PARITY_MAX_ABS_DIFF = 1e-9
COVERAGE_MIN = 0.75

# SHA-256 and row counts for the 2023 files. 2024 stays in ride_pricing.config
# so the parent pins remain the single source for those four months.
EXTRA_SHA256: dict[str, str] = {
    "2023-05": "9bd7d1c557bb9d0413619ba296b4828b884852a70bd07f2230b83f080d9a8591",
    "2023-06": "3e60e29b5df45683948b68acbfb503aa459d14162c610622585c6580fdbfc73a",
    "2023-07": "98504ae351d920e9e0b61241f5707f90ee1588cfb199c378459234b5035805ab",
    "2023-08": "84ce36a2f84fa14c12fd976c79a7d4f30c871f1782b08ca7c27bb76ab946e66b",
    "2023-09": "091bf156bea5bb10bd217f5f12375f966f8789349af45d5ee4153e3e1e077a87",
    "2023-10": "dc5f069476d350df802ed8120f3b39ec321c34cfa905381e87f608c78ca0c645",
    "2023-11": "af203de0e73a92d640121f02a34326f25037d62380fc406237c87c2a4231ae58",
    "2023-12": "ab4f4db11cb31484747f62eb507c13fcfaff290aaf59c13e6465dea0b1166b9c",
}
EXTRA_ROWS: dict[str, int] = {
    "2023-05": 3513649,
    "2023-06": 3307234,
    "2023-07": 2907108,
    "2023-08": 2824209,
    "2023-09": 2846722,
    "2023-10": 3522285,
    "2023-11": 3339715,
    "2023-12": 3376567,
}


def pins_for(month: str) -> tuple[str, int]:
    if month in TLC_SHA256:
        return TLC_SHA256[month], int(TLC_ROWS[month])
    digest = EXTRA_SHA256.get(month, "")
    rows = int(EXTRA_ROWS.get(month, 0))
    if len(digest) != 64 or rows <= 0:
        raise RuntimeError(
            f"checksum for {month} is not pinned; record sha256 and row count before a full run"
        )
    return digest, rows


def all_pins() -> dict[str, int]:
    return {month: pins_for(month)[1] for month in SPARK_MONTHS}
