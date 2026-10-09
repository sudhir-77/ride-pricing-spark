import json

from ride_pricing.reporting import render_extension_notes


def test_extension_notes_use_metric_digits(tmp_path):
    payload = {
        "metrics": {
            "spark": {
                "total_trips": 38123456,
                "n_months": 12,
                "parity_match_rate": 1.0,
                "runtime_s": 12.5,
            },
            "demand_11m": {
                "april_top50_wape": 0.141234,
                "april_top50_wape_seasonal_naive": 0.192966,
                "n_history_months": 11,
            },
            "airflow": {"n_tasks": 5, "all_succeeded": True},
            "forecast": {"april_top50_wape_lightgbm": 0.146127},
        }
    }
    text = render_extension_notes(payload)
    assert "38123456" in text
    assert "0.1412" in text
    assert "0.1930" in text
    assert "0.1461" in text
    assert "1.0000" in text
    lowered = text.lower()
    for banned in ("leveraged", "robust", "cutting-edge", "seamless", "spearheaded"):
        assert banned not in lowered
    assert "\u2014" not in text
    notes = tmp_path / "RESUME_NOTES.md"
    notes.write_text(text, encoding="utf-8")
    assert json.dumps(payload["metrics"]["spark"]["total_trips"]) in notes.read_text(encoding="utf-8")
