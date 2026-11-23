"""Pandera schemas for the common analytical representation.

Schemas enforce *conformance* only: column presence, types, identifier format,
flag-code format and the "no silent nulling" rule. Plausibility (ranges,
temporal ordering, duplicates) is deliberately left to quality assessment, so
that real-world data problems are reported rather than blocking ingestion or
being silently fixed.

All timestamps in one table share one temporal policy: either naive local clock
time (``tz=None``) or timezone-aware time in a single zone (e.g. ``"UTC"``).
"""

from __future__ import annotations

import pandas as pd
import pandera.pandas as pa

from .annotations import FLAG_CODE_PATTERN

SCHEMA_VERSION = "0.1"


def _is_flag_list(value) -> bool:
    return isinstance(value, list) and all(isinstance(c, str) and FLAG_CODE_PATTERN.match(c) for c in value)


def _flags_column() -> pa.Column:
    return pa.Column(
        object,
        nullable=False,
        checks=pa.Check(lambda s: s.map(_is_flag_list), error="quality_flags must be lists of lower_snake_case codes"),
    )


def _timestamp_dtype(tz: str | None):
    return "datetime64[ns]" if tz is None else pd.DatetimeTZDtype(unit="ns", tz=tz)


def _non_empty_str(nullable: bool = False) -> pa.Column:
    return pa.Column(str, nullable=nullable, checks=pa.Check.str_length(min_value=1))


def _lineage_columns(with_column: bool = True) -> dict[str, pa.Column]:
    cols = {
        "source_file": _non_empty_str(),
        "source_row": pa.Column("Int64", nullable=True),
    }
    if with_column:
        cols["source_column"] = pa.Column(str, nullable=True)
    return cols


def _participant_prefix_check() -> pa.Check:
    """participant_id must be namespaced by its source: '<source_dataset>:<local id>'."""

    def check(df: pd.DataFrame) -> pd.Series:
        return pd.Series(
            [isinstance(p, str) and isinstance(s, str) and p.startswith(s + ":") and len(p) > len(s) + 1
             for p, s in zip(df["participant_id"], df["source_dataset"])],
            index=df.index,
        )

    return pa.Check(check, error="participant_id must be '<source_dataset>:<id>'")


def _no_silent_nulling_check(value_columns: list[str]) -> pa.Check:
    """A row whose raw value exists but whose standardised value is null must carry a flag."""

    def check(df: pd.DataFrame) -> pd.Series:
        nulled = df["raw_value"].notna() & df[value_columns].isna().all(axis=1)
        flagged = df["quality_flags"].map(len) > 0
        return ~nulled | flagged

    return pa.Check(check, error="value nulled without a quality flag (silent nulling is not allowed)")


def observation_schema(tz: str | None = None) -> pa.DataFrameSchema:
    """Time-stamped measurements, one row per stream reading."""
    return pa.DataFrameSchema(
        columns={
            "participant_id": _non_empty_str(),
            "source_dataset": _non_empty_str(),
            "stream_id": _non_empty_str(),
            "concept": _non_empty_str(),
            "timestamp": pa.Column(_timestamp_dtype(tz), nullable=False),
            "value": pa.Column(float, nullable=True),
            "unit": pa.Column(str, nullable=True),
            "raw_value": pa.Column(str, nullable=True),
            "is_interpolated": pa.Column(bool, nullable=False),
            "quality_flags": _flags_column(),
            **_lineage_columns(),
        },
        checks=[_participant_prefix_check(), _no_silent_nulling_check(["value"])],
        strict=True,
        ordered=False,
        name="observations",
        metadata={"pghd_table": "observations", "schema_version": SCHEMA_VERSION, "tz": tz},
    )


def event_schema(tz: str | None = None) -> pa.DataFrameSchema:
    """Discrete, time-anchored occurrences. Condition modules extend this with typed attributes."""
    ts = _timestamp_dtype(tz)
    return pa.DataFrameSchema(
        columns={
            "participant_id": _non_empty_str(),
            "source_dataset": _non_empty_str(),
            "event_id": pa.Column(str, nullable=False, unique=True, checks=pa.Check.str_length(min_value=1)),
            "event_type": _non_empty_str(),
            "event_subtype": pa.Column(str, nullable=True),
            "raw_label": pa.Column(str, nullable=True),
            "start_time": pa.Column(ts, nullable=False),
            "end_time": pa.Column(ts, nullable=True),
            "quality_flags": _flags_column(),
            **_lineage_columns(with_column=False),
        },
        checks=[_participant_prefix_check()],
        strict=True,
        ordered=False,
        name="events",
        metadata={"pghd_table": "events", "schema_version": SCHEMA_VERSION, "tz": tz},
    )


def participant_attribute_schema(tz: str | None = None) -> pa.DataFrameSchema:
    """Participant-level or baseline values in long format."""
    return pa.DataFrameSchema(
        columns={
            "participant_id": _non_empty_str(),
            "source_dataset": _non_empty_str(),
            "concept": _non_empty_str(),
            "value_num": pa.Column(float, nullable=True),
            "value_text": pa.Column(str, nullable=True),
            "unit": pa.Column(str, nullable=True),
            "raw_value": pa.Column(str, nullable=True),
            "measured_at": pa.Column(_timestamp_dtype(tz), nullable=True),
            # original date/time text when it cannot be resolved to a full timestamp (e.g. clock time only)
            "measured_at_raw": pa.Column(str, nullable=True),
            "quality_flags": _flags_column(),
            **_lineage_columns(),
        },
        checks=[_participant_prefix_check(), _no_silent_nulling_check(["value_num", "value_text"])],
        strict=True,
        ordered=False,
        name="participant_attributes",
        metadata={"pghd_table": "participant_attributes", "schema_version": SCHEMA_VERSION, "tz": tz},
    )


def stream_schema(tz: str | None = None) -> pa.DataFrameSchema:
    """One row per participant x stream: what was recorded and how it was processed."""
    ts = _timestamp_dtype(tz)
    return pa.DataFrameSchema(
        columns={
            "participant_id": _non_empty_str(),
            "source_dataset": _non_empty_str(),
            "stream_id": _non_empty_str(),
            "concept": _non_empty_str(),
            "device": pa.Column(str, nullable=True),
            "declared_native_interval_s": pa.Column(float, nullable=True, checks=pa.Check.gt(0)),
            "first_timestamp": pa.Column(ts, nullable=True),
            "last_timestamp": pa.Column(ts, nullable=True),
            "n_rows_source": pa.Column("Int64", nullable=False, checks=pa.Check.ge(0)),
            "n_rows_output": pa.Column("Int64", nullable=False, checks=pa.Check.ge(0)),
            "quality_flags": _flags_column(),
        },
        checks=[_participant_prefix_check()],
        unique=["participant_id", "stream_id"],
        strict=True,
        ordered=False,
        name="streams",
        metadata={"pghd_table": "streams", "schema_version": SCHEMA_VERSION, "tz": tz},
    )


TABLE_SCHEMAS = {
    "observations": observation_schema,
    "events": event_schema,
    "participant_attributes": participant_attribute_schema,
    "streams": stream_schema,
}
