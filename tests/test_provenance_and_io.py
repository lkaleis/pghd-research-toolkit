import json
from datetime import datetime, timezone

import pandas as pd
import pandera.errors as pe
import pytest

from pghd_ready.core.annotations import AnnotationError
from pghd_ready.core.io import read_metadata, read_table, write_table
from pghd_ready.core.provenance import RunManifest, TransformLog, config_hash, sha256_file
from pghd_ready.core.schemas import (
    event_schema,
    observation_schema,
    participant_attribute_schema,
    stream_schema,
)

from conftest import make_attributes, make_events, make_observations, make_streams


def fixed_clock():
    return datetime(2000, 1, 1, tzinfo=timezone.utc)


# ------------------------------------------------------------------ provenance
def test_config_hash_ignores_key_order():
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_transform_log_enforces_basis_rules(annotation_registry, tmp_path):
    log = TransformLog(annotation_registry, clock=fixed_clock)
    log.record("rename", "core.rename", "structural", 10, 10)
    log.record("null_error_codes", "core.rules", "documented", 10, 10,
               evidence="synthetic dictionary", flags_raised={"documented_error_code": 1})
    log.record("keep_lattice", "adapter.recover", "assumption", 10, 4,
               assumption_ids=["A-SYN-01"], flags_raised={"lattice_reading": 4})
    with pytest.raises(AnnotationError, match="assumption_ids"):
        log.record("x", "f", "assumption", 1, 1)
    with pytest.raises(AnnotationError, match="evidence"):
        log.record("x", "f", "documented", 1, 1)
    with pytest.raises(AnnotationError, match="unknown assumptions"):
        log.record("x", "f", "assumption", 1, 1, assumption_ids=["A-NOPE"])
    with pytest.raises(AnnotationError, match="unregistered flags"):
        log.record("x", "f", "structural", 1, 1, flags_raised={"made_up": 1})
    with pytest.raises(AnnotationError, match="basis is 'structural'"):
        log.record("x", "f", "structural", 1, 1, assumption_ids=["A-SYN-01"])

    out = tmp_path / "log.jsonl"
    log.write_jsonl(out)
    lines = [json.loads(l) for l in out.read_text().splitlines()]
    assert [l["basis"] for l in lines] == ["structural", "documented", "assumption"]
    assert lines[2]["rows_out"] == 4


def test_manifest_is_deterministic_and_relative(tmp_path, annotation_registry):
    (tmp_path / "raw").mkdir()
    f = tmp_path / "raw" / "input.csv"
    f.write_text("a,b\n1,2\n")

    def build():
        m = RunManifest("test", tmp_path, clock=fixed_clock, record_git=False)
        m.add_input(f)
        m.add_config("source", {"x": 1})
        m.set_source({"timezone": None, "date_shifted": True})
        m.set_annotations(annotation_registry)
        return m

    a, b = build().to_dict(), build().to_dict()
    assert a == b
    assert a["inputs"][0]["path"] == "raw/input.csv"
    assert a["inputs"][0]["sha256"] == sha256_file(f)
    assert "documented_error_code" in a["annotations"]["flags"]

    outside = tmp_path.parent / "outside.csv"
    outside.write_text("x")
    try:
        with pytest.raises(ValueError, match="outside"):
            build().add_input(outside)
    finally:
        outside.unlink()


# ------------------------------------------------------------------ io
@pytest.mark.parametrize(
    "schema_fn,factory,name",
    [
        (observation_schema, make_observations, "observations"),
        (event_schema, make_events, "events"),
        (participant_attribute_schema, make_attributes, "participant_attributes"),
        (stream_schema, make_streams, "streams"),
    ],
)
@pytest.mark.parametrize("tz", [None, "UTC"])
def test_parquet_round_trip(tmp_path, schema_fn, factory, name, tz):
    df = factory(tz)
    path = tmp_path / f"{name}.parquet"
    write_table(df, schema_fn(tz), path)
    back = read_table(path, schema_fn(tz))
    pd.testing.assert_frame_equal(back.reset_index(drop=True), df.reset_index(drop=True), check_like=True)
    assert read_metadata(path)["pghd_table"] == name


def test_round_trip_with_all_empty_flag_lists(tmp_path):
    df = make_observations()
    df["quality_flags"] = [[] for _ in range(len(df))]
    df.loc[2, "raw_value"] = None  # keep the no-silent-nulling rule satisfied
    path = tmp_path / "obs.parquet"
    write_table(df, observation_schema(), path)
    assert read_table(path, observation_schema())["quality_flags"].tolist() == [[], [], []]


def test_invalid_table_is_not_written(tmp_path):
    df = make_observations()
    df["participant_id"] = "no_namespace"
    path = tmp_path / "bad.parquet"
    with pytest.raises(pe.SchemaErrors):
        write_table(df, observation_schema(), path)
    assert not path.exists()


def test_reading_wrong_table_type_fails(tmp_path):
    path = tmp_path / "events.parquet"
    write_table(make_events(), event_schema(), path)
    with pytest.raises(ValueError, match="expected 'observations'"):
        read_table(path, observation_schema())
