"""Shared synthetic fixtures. No real participant data is used anywhere in the test suite."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from pghd_ready.core.annotations import AnnotationRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = "synthetic"


def make_observations(tz: str | None = None) -> pd.DataFrame:
    ts = pd.date_range("2000-01-01 08:00", periods=3, freq="5min", tz=tz, unit="ns")
    return pd.DataFrame(
        {
            "participant_id": [f"{SRC}:p1"] * 3,
            "source_dataset": [SRC] * 3,
            "stream_id": ["cgm_a"] * 3,
            "concept": ["glucose_interstitial"] * 3,
            "timestamp": ts,
            "value": [100.0, 105.5, None],
            "unit": ["mg/dL"] * 3,
            "raw_value": ["100", "105.5", None],
            "is_interpolated": [False, False, False],
            "quality_flags": [[], ["suspected_spike"], []],
            "source_file": ["export/p1.csv"] * 3,
            "source_row": pd.array([1, 2, 3], dtype="Int64"),
            "source_column": ["glucose"] * 3,
        }
    )


def make_events(tz: str | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": [f"{SRC}:p1", f"{SRC}:p1"],
            "source_dataset": [SRC, SRC],
            "event_id": ["synthetic:p1:e1", "synthetic:p1:e2"],
            "event_type": ["meal", "meal"],
            "event_subtype": ["breakfast", None],
            "raw_label": ["Breakfast", "brunch?"],
            "start_time": pd.to_datetime(["2000-01-01 08:00", "2000-01-01 11:00"]).tz_localize(tz).as_unit("ns"),
            "end_time": pd.to_datetime([None, None]).tz_localize(tz).as_unit("ns"),
            "raw_attributes": [None, None],
            "quality_flags": [[], ["unmapped_meal_label"]],
            "source_file": ["export/food.csv"] * 2,
            "source_row": pd.array([1, 2], dtype="Int64"),
        }
    )


def make_attributes(tz: str | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": [f"{SRC}:p1"] * 3,
            "source_dataset": [SRC] * 3,
            "concept": ["body_weight", "sex", "ldl_cholesterol_calculated"],
            "value_num": [70.0, None, None],
            "value_text": [None, "F", None],
            "unit": ["kg", None, "mg/dL"],
            "raw_value": ["154.3", "F", "999"],
            "measured_at": pd.to_datetime([None, None, None]).tz_localize(tz).as_unit("ns"),
            "measured_at_raw": [None, None, "9:15"],
            "quality_flags": [[], [], ["documented_error_code"]],
            "source_file": ["export/profile.csv"] * 3,
            "source_row": pd.array([1, 1, 1], dtype="Int64"),
            "source_column": ["weight_lb", "sex", "ldl"],
        }
    )


def make_streams(tz: str | None = None) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "participant_id": [f"{SRC}:p1", f"{SRC}:p1"],
            "source_dataset": [SRC, SRC],
            "stream_id": ["cgm_a", "cgm_b"],
            "concept": ["glucose_interstitial", "glucose_interstitial"],
            "device": ["device A", "device B"],
            "declared_native_interval_s": [300.0, 900.0],
            "first_timestamp": pd.to_datetime(["2000-01-01", "2000-01-01"]).tz_localize(tz).as_unit("ns"),
            "last_timestamp": pd.to_datetime(["2000-01-10", "2000-01-12"]).tz_localize(tz).as_unit("ns"),
            "n_rows_source": pd.array([10, 20], dtype="Int64"),
            "n_rows_output": pd.array([10, 5], dtype="Int64"),
            "quality_flags": [[], []],
        }
    )


SYNTHETIC_ANNOTATIONS = {
    "assumptions": {
        "A-SYN-01": {
            "statement": "Readings on a fixed lattice are the device's native samples.",
            "rationale": "Synthetic test assumption.",
        }
    },
    "flags": {
        "suspected_spike": {"category": "suspected_anomaly", "description": "Implausible jump between readings."},
        "unmapped_meal_label": {"category": "semantic_ambiguity", "description": "Meal label not in the mapping."},
        "documented_error_code": {
            "category": "documented_issue",
            "description": "Value documented by the source as an error code.",
            "action": "set_null",
            "evidence": "synthetic data dictionary, field 'ldl'",
        },
        "lattice_reading": {
            "category": "assumption_based",
            "description": "Value retained as a native reading under A-SYN-01.",
            "assumption_id": "A-SYN-01",
        },
    },
}


@pytest.fixture
def annotation_registry() -> AnnotationRegistry:
    return AnnotationRegistry.from_mapping(SYNTHETIC_ANNOTATIONS)
