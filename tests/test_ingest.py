"""End-to-end tests of the generic ingestion engine on the persisted synthetic demo dataset."""

import json

import numpy as np
import pandas as pd
import pytest

from pghd_ready.conditions.cardiometabolic.schemas import meal_event_schema
from pghd_ready.core.config import ConfigError, SourceConfig
from pghd_ready.core.ingest import IngestError, ingest_source, write_outputs
from pghd_ready.core.io import read_table
from pghd_ready.core.schemas import observation_schema, participant_attribute_schema, rejected_record_schema
from pghd_ready.core.transforms import TransformError

import synthetic_sources as syn
from synthetic_sources import APP, DEVICE, PID, PROFILE

TZ = "UTC"


@pytest.fixture
def wide(tmp_path):
    root = syn.write_wide_source(tmp_path / "raw")
    cfg_path = syn.write_config(tmp_path / "source.yaml", syn.wide_config())
    return root, SourceConfig.from_yaml(cfg_path)


@pytest.fixture
def result(wide):
    root, cfg = wide
    return ingest_source(cfg, root, syn.concepts())


def _obs(result, stream, pid):
    o = result.observations
    return o[(o.stream_id == stream) & (o.participant_id == PID.format(pid))].sort_values("source_row").reset_index(drop=True)


def _rej(result, file, line):
    r = result.rejected_records
    return r[(r.source_file == file) & (r.source_row == line)].iloc[0]


# ---------------------------------------------------------------- structure handling
def test_participants_are_namespaced_and_padded(result):
    assert sorted(result.observations.participant_id.unique()) == [PID.format(i) for i in (1, 2, 3)]
    assert sorted(result.participant_attributes.participant_id.unique()) == [PID.format(i) for i in (1, 2, 3, 4)]


def test_reordered_and_aliased_columns_are_resolved_by_name(result):
    assert _obs(result, "patch_a", 2).value.tolist() == [95.0, 96.0, 97.0]
    assert _obs(result, "wrist_hr", 2).source_column.unique().tolist() == ["Heart Rate"]
    assert _obs(result, "patch_b", 1).source_column.unique().tolist() == ["Sensor B "]


def test_missing_optional_columns_are_recorded_not_guessed(result):
    res = result.column_resolution
    f2 = "device_export/user_02/user_02.csv"
    assert res[(res.file == f2) & (res.role == "observation:wrist_met")].status.tolist() == ["missing_optional"]
    assert len(_obs(result, "wrist_met", 2)) == 0
    assert res[(res.file == f2) & (res.role == "event:meal.attribute:amount_consumed_raw")].status.tolist() == [
        "missing_optional"]


def test_unused_and_dropped_columns_are_reported(result):
    fs = result.file_stats.set_index("source_file")
    assert fs.loc["device_export/user_01/user_01.csv", "unused_columns"] == "Note"
    assert fs.loc["device_export/user_01/user_01.csv", "dropped_columns"] == "Unnamed: 0"
    assert fs.loc["profile.csv", "unused_columns"] == "Time"


# ---------------------------------------------------------------- physical line traceability
def test_source_row_is_physical_line(result):
    # user_01: header line 1; records start on lines 2, 3, 4 (quoted note spans 4-5), 6; blank line 7
    assert _obs(result, "patch_a", 1).source_row.tolist() == [2, 3, 4, 6]
    assert _obs(result, "patch_b", 1).source_row.tolist() == [2, 6]
    fs = result.file_stats.set_index("source_file").loc["device_export/user_01/user_01.csv"]
    assert (fs.n_physical_lines, fs.n_records, fs.n_blank_lines) == (9, 6, 1)


def test_source_row_points_at_the_right_text(result, wide):
    root, _ = wide
    lines = (root / "device_export/user_03/user_03.csv").read_text().splitlines()
    for r in _obs(result, "patch_a", 3).itertuples():
        assert lines[r.source_row - 1].split(",")[1] == r.raw_value


# ---------------------------------------------------------------- rejected records keep their content
def test_rejected_records_preserve_content_and_reason(result):
    rej = result.rejected_records
    rejected_record_schema(TZ).validate(rej, lazy=True)
    assert sorted(rej.reason.tolist()) == sorted([
        "missing_timestamp", "field_count_mismatch", "timestamp_dst_unresolvable", "unparseable_timestamp",
        "missing_timestamp", "missing_participant_id", "empty_row"])
    miss = _rej(result, "device_export/user_01/user_01.csv", 8)
    assert miss.reason == "missing_timestamp" and miss.participant_id == PID.format(1)
    content = json.loads(miss.raw_record)
    assert content["Note"] == "missing timestamp" and content["Sensor A"] == "104"
    assert content["Unnamed: 0"] == "4"  # dropped column is still preserved in the audit record
    over = _rej(result, "device_export/user_01/user_01.csv", 9)
    assert over.reason == "field_count_mismatch" and len(json.loads(over.raw_record)["fields"]) == 13
    assert json.loads(_rej(result, "device_export/user_03/user_03.csv", 3).raw_record)["Timestamp"] == "not a time"
    assert _rej(result, "device_export/user_02/user_02.csv", 5).reason == "timestamp_dst_unresolvable"
    nopid = _rej(result, "app_export/app_glucose.csv", 9)
    assert nopid.reason == "missing_participant_id" and nopid.participant_id is None
    assert _rej(result, "profile.csv", 4).reason == "empty_row"
    assert (rej.record_scope == "source_row").all()


def test_short_rows_are_padded_and_counted(result):
    fs = result.file_stats.set_index("source_file").loc["device_export/user_03/user_03.csv"]
    assert fs.n_short_rows_padded == 1
    assert _obs(result, "patch_b", 3).raw_value.tolist() == ["120", "121", "122"]


# ---------------------------------------------------------------- values and annotations
def test_unparseable_value_kept_raw_and_flagged(result):
    row = _obs(result, "patch_a", 3).iloc[0]
    assert row.raw_value == "HIGH" and np.isnan(row.value) and row.quality_flags == ["unparseable_numeric"]


def test_scale_is_applied_and_logged_with_documented_basis(result):
    assert _obs(result, "wrist_met", 1).value.tolist() == pytest.approx([1.0, 1.2, 1.5, 1.1])
    step = next(r for r in result.transform_log.records if r.operation == "scale")
    assert step.basis.value == "documented" and step.evidence and step.target == "wrist_met"


def test_timezone_localisation_is_assumption_based(result):
    assert str(result.observations.timestamp.dt.tz) == "UTC"
    assert _obs(result, "patch_a", 1).timestamp.iloc[0] == pd.Timestamp("2000-01-01 07:00", tz="UTC")  # 08:00 CET
    step = next(r for r in result.transform_log.records if r.operation == "timestamps.localize")
    assert step.basis.value == "assumption" and step.assumption_ids == ["A-DEMO-TZ-01"]
    assert step.rows_in - step.rows_out == 1


def test_static_semantic_flag_on_undocumented_stream(result):
    lvl = _obs(result, "wrist_intensity", 2)
    assert lvl.value.tolist() == [0.0, 1.0, 3.0] and lvl.unit.isna().all()
    assert all(f == ["semantics_undocumented"] for f in lvl.quality_flags)


def test_attribute_unit_conversion_error_codes_and_unparseable(result):
    att = result.participant_attributes.set_index(["participant_id", "concept"])
    w = att.loc[(PID.format(1), "body_weight")]
    assert w.value_num == pytest.approx(154.32 * 0.45359237) and w.unit == "kg" and w.raw_value == "154.32"
    ldl = att.loc[(PID.format(1), "ldl_cholesterol_calculated")]
    assert np.isnan(ldl.value_num) and ldl.raw_value == "999" and ldl.quality_flags == ["documented_error_code"]
    kg = att.loc[(PID.format(4), "body_weight")]
    assert np.isnan(kg.value_num) and kg.raw_value == "90 kg" and kg.quality_flags == ["unparseable_numeric"]
    rule = next(r for r in result.transform_log.records if r.operation == "rule" and r.target == "ldl_cholesterol_calculated")
    assert rule.basis.value == "documented" and rule.flags_raised == {"documented_error_code": 1}


def test_categorical_values_preserved_and_duplicate_headers(result):
    att = result.participant_attributes
    assert sorted(att[att.concept == "sex"].value_text) == ["F", "F", "M", "X"]
    fs = att[att.concept == "capillary_glucose"].sort_values("participant_id")
    assert fs.measured_at_raw.tolist() == ["9:15", "10:05", "9:40"]
    assert fs.measured_at.isna().all()  # date unknown: no timestamp is invented


def test_long_table_in_demo_uses_offsets_units_and_filters(result):
    app = result.observations[result.observations.stream_id == "app_glucose"]
    p1 = app[app.participant_id == PID.format(1)].sort_values("timestamp")
    assert p1.timestamp.dt.strftime("%H:%M").tolist() == ["07:05", "07:10"]  # Z and +01:00 both to UTC
    assert p1.value.tolist() == pytest.approx([5.6 * 18.016, 5.9 * 18.016])
    uns = result.unselected_values
    assert sorted(uns.value) == ["Glucose", "steps"] and set(uns.column) == {"type"}


# ---------------------------------------------------------------- events
def test_events_mapping_ambiguity_and_raw_attributes(result):
    ev = result.events
    meal_event_schema(TZ).validate(ev, lazy=True)
    meals = ev[ev.event_type == "meal"].set_index("raw_label")
    assert meals.loc["Breakfast", "event_subtype"] == "breakfast"
    assert meals.loc["brunch", "event_subtype"] is None
    assert meals.loc["brunch", "quality_flags"] == ["unmapped_category"]
    assert meals.loc["snack", "amount_consumed_raw"] == 300.0
    assert meals.loc["snack", "quality_flags"] == ["amount_outside_documented_range"]
    dinner = meals.loc["Dinner"]
    assert np.isnan(dinner.energy_kcal) and dinner.quality_flags == ["unparseable_numeric"]
    assert json.loads(dinner.raw_attributes)["energy_kcal"] == "about 700"
    assert json.loads(meals.loc["Breakfast", "raw_attributes"]) == {
        "energy_kcal": "400", "carbohydrate_g": "50", "amount_consumed_raw": "100", "image_ref": "photos/a.jpg"}
    assert (ev.event_type == "meal_photo").sum() == 3
    assert ev.event_id.is_unique and all(i.endswith(f":{r}") for i, r in zip(ev.event_id, ev.source_row))


def test_streams_table(result):
    st = result.streams.set_index(["participant_id", "stream_id"])
    a1 = st.loc[(PID.format(1), "patch_a")]
    assert (a1.n_rows_source, a1.n_rows_output, a1.device, a1.declared_native_interval_s) == (
        4, 4, "Patch sensor A (fictional)", 60.0)


def test_ingestion_is_deterministic(wide):
    root, cfg = wide
    a = ingest_source(cfg, root, syn.concepts())
    b = ingest_source(cfg, root, syn.concepts())
    for name in ("observations", "events", "participant_attributes", "streams", "rejected_records", "file_stats"):
        pd.testing.assert_frame_equal(getattr(a, name), getattr(b, name))


# ---------------------------------------------------------------- outputs and provenance
def test_write_outputs_round_trip_and_manifest(result, tmp_path):
    out = tmp_path / "processed"
    manifest = write_outputs(result, out, base_dir=tmp_path, record_git=False)
    pd.testing.assert_frame_equal(read_table(out / "observations.parquet", observation_schema(TZ)), result.observations)
    read_table(out / "participant_attributes.parquet", participant_attribute_schema(TZ))
    read_table(out / "events.parquet", meal_event_schema(TZ))
    pd.testing.assert_frame_equal(read_table(out / "rejected_records.parquet", rejected_record_schema(TZ)),
                                  result.rejected_records)
    assert len(manifest["inputs"]) == 5 and all(not i["path"].startswith("/") for i in manifest["inputs"])
    log = [json.loads(l) for l in (out / "_provenance" / "transform_log.jsonl").read_text().splitlines()]
    assert {l["basis"] for l in log} == {"structural", "documented", "assumption"}
    assert all(l["operation"] for l in log)
    summary = (out / "ingest_summary.json").read_text()
    assert "HIGH" not in summary and "user_0" not in summary  # aggregate only


# ---------------------------------------------------------------- failure modes
def test_missing_required_column_fails(tmp_path):
    root = syn.write_wide_source(tmp_path / "raw")
    (root / "device_export/user_03/user_03.csv").write_text("Timestamp,Sensor B\n2000-01-01 00:00:00,1\n")
    with pytest.raises(IngestError, match="required column"):
        ingest_source(SourceConfig.from_mapping(syn.wide_config()), root, syn.concepts())


def test_ambiguous_alias_match_fails(tmp_path):
    root = syn.write_wide_source(tmp_path / "raw")
    (root / "device_export/user_02/user_02.csv").write_text(
        "Timestamp,Sensor A,HR,Heart Rate\n2000-01-01 00:00:00,1,60,61\n")
    with pytest.raises(IngestError, match="matches several columns"):
        ingest_source(SourceConfig.from_mapping(syn.wide_config()), root, syn.concepts())


def test_unregistered_flag_in_config_fails(tmp_path):
    root = syn.write_wide_source(tmp_path / "raw")
    cfg = syn.wide_config()
    cfg["tables"][PROFILE]["attributes"][3]["rules"][0]["flag"] = "made_up_flag"
    with pytest.raises(ConfigError, match="unregistered flag"):
        ingest_source(SourceConfig.from_mapping(cfg), root, syn.concepts())


def test_numeric_concept_requires_convertible_source_unit(tmp_path):
    root = syn.write_wide_source(tmp_path / "raw")
    cfg = syn.wide_config()
    cfg["tables"][PROFILE]["attributes"][0]["unit"] = "mmol/L"
    with pytest.raises(Exception, match="no defined conversion"):
        ingest_source(SourceConfig.from_mapping(cfg), root, syn.concepts())
    cfg["tables"][PROFILE]["attributes"][0].pop("unit")
    with pytest.raises(ConfigError, match="requires a source 'unit'"):
        ingest_source(SourceConfig.from_mapping(cfg), root, syn.concepts())


# ---------------------------------------------------------------- focused timezone cases
def test_long_format_with_offsets_to_utc(tmp_path):
    root = syn.write_long_source(tmp_path / "raw", with_offsets=True)
    res = ingest_source(SourceConfig.from_mapping(syn.long_config({"column": "recorded_at", "format": "ISO8601"})),
                        root, syn.concepts())
    u1 = res.observations[res.observations.participant_id == "synthetic_app:u1"].sort_values("timestamp")
    assert len(res.observations) == 3
    assert u1.timestamp.dt.strftime("%H:%M").tolist() == ["00:30", "01:30"]
    assert u1.raw_value.tolist() == ["5.5", "6.0"]


def test_naive_times_localised_only_when_declared_with_basis(tmp_path):
    root = syn.write_long_source(tmp_path / "raw", with_offsets=False)
    ts = {"column": "recorded_at", "format": "%Y-%m-%d %H:%M:%S"}
    with pytest.raises(IngestError, match="assume_timezone"):
        ingest_source(SourceConfig.from_mapping(syn.long_config(ts)), root, syn.concepts())
    with pytest.raises(ConfigError, match="requires a 'basis'"):
        SourceConfig.from_mapping(syn.long_config({**ts, "assume_timezone": "Europe/Berlin"}))
    cfg = syn.long_config({**ts, "assume_timezone": "Europe/Berlin", "basis": "assumption", "assumption_ids": ["A-TZ"]})
    res = ingest_source(SourceConfig.from_mapping(cfg), root, syn.concepts())
    assert len(res.observations) == 2  # the nonexistent 02:30 local time is rejected, not shifted
    assert res.rejected_records.reason.tolist() == ["timestamp_dst_unresolvable"]
    assert res.rejected_records.source_row.tolist() == [3]


def test_offsets_with_naive_output_policy_fail(tmp_path):
    root = syn.write_long_source(tmp_path / "raw", with_offsets=True)
    cfg = syn.long_config({"column": "recorded_at", "format": "ISO8601"})
    cfg["temporal"]["output_timezone"] = None
    with pytest.raises(IngestError, match="output_timezone is null"):
        ingest_source(SourceConfig.from_mapping(cfg), root, syn.concepts())


# ---------------------------------------------------------------- transforms and the guard
def _with_transform(transform: dict) -> SourceConfig:
    cfg = syn.wide_config(plugins=["synthetic_plugin"])
    cfg["tables"][DEVICE]["observations"][0]["transforms"] = [transform]
    return SourceConfig.from_mapping(cfg)


def _run(tmp_path, transform):
    return ingest_source(_with_transform(transform), syn.write_wide_source(tmp_path / "raw"), syn.concepts())


KEEP = {"name": "synthetic.keep_every_other", "params": {"step": 2}, "basis": "assumption",
        "assumption_ids": ["A-SYN-01"], "flags": ["lattice_reading"]}


def test_assumption_based_transform_is_logged_and_removed_rows_are_kept(tmp_path):
    res = _run(tmp_path, KEEP)
    a1 = _obs(res, "patch_a", 1)
    assert a1.source_row.tolist() == [2, 4]
    assert all("lattice_reading" in f for f in a1.quality_flags)
    rec = next(r for r in res.transform_log.records if r.operation == "transform")
    assert rec.assumption_ids == ["A-SYN-01"] and rec.rows_in == 10 and rec.rows_out == 6
    removed = res.rejected_records[res.rejected_records.reason == "removed_by_transform"]
    assert len(removed) == 4 and (removed.record_scope == "observation").all()
    assert json.loads(removed.iloc[0].raw_record)["stream_id"] == "patch_a"
    s1 = res.streams.set_index(["participant_id", "stream_id"]).loc[(PID.format(1), "patch_a")]
    assert (s1.n_rows_source, s1.n_rows_output) == (4, 2)


def test_value_change_with_declared_explanatory_flag_is_allowed(tmp_path):
    res = _run(tmp_path, {"name": "synthetic.change_with_flag", "params": {"flag": "lattice_reading"},
                          "basis": "assumption", "assumption_ids": ["A-SYN-01"], "flags": ["lattice_reading"]})
    assert _obs(res, "patch_a", 1).value.tolist() == [101.0, 102.0, 103.0, 104.0]


def test_silent_value_change_is_rejected(tmp_path):
    with pytest.raises(IngestError, match="changed values without adding an explanatory flag"):
        _run(tmp_path, {"name": "synthetic.silent_change", "basis": "structural"})


def test_unrelated_registered_flag_cannot_bypass_the_guard(tmp_path):
    # the flag is registered, but the transform did not declare it
    with pytest.raises(IngestError, match="not declared in its 'flags'"):
        _run(tmp_path, {"name": "synthetic.change_with_flag", "params": {"flag": "semantics_undocumented"},
                        "basis": "assumption", "assumption_ids": ["A-SYN-01"], "flags": ["lattice_reading"]})


def test_declared_but_non_explanatory_flag_cannot_justify_a_change(tmp_path):
    # semantic-ambiguity flags annotate; they cannot explain a changed value
    with pytest.raises(IngestError, match="changed values without adding an explanatory flag"):
        _run(tmp_path, {"name": "synthetic.change_with_flag", "params": {"flag": "semantics_undocumented"},
                        "basis": "assumption", "assumption_ids": ["A-SYN-01"],
                        "flags": ["lattice_reading", "semantics_undocumented"]})


def test_structural_transform_cannot_change_values_even_with_assumption_flag(tmp_path):
    with pytest.raises(ConfigError, match="does not declare in assumption_ids"):
        _run(tmp_path, {"name": "synthetic.change_with_flag", "params": {"flag": "lattice_reading"},
                        "basis": "structural", "flags": ["lattice_reading"]})


def test_flag_of_another_assumption_is_rejected_at_configuration(tmp_path):
    with pytest.raises(ConfigError, match="relies on assumption A-SYN-02"):
        _run(tmp_path, {"name": "synthetic.change_with_flag", "params": {"flag": "other_assumption_flag"},
                        "basis": "assumption", "assumption_ids": ["A-SYN-01"], "flags": ["other_assumption_flag"]})


def test_protected_columns_cannot_be_modified(tmp_path):
    with pytest.raises(IngestError, match="protected column 'raw_value'"):
        _run(tmp_path, {**KEEP, "name": "synthetic.tamper_raw"})


def test_new_rows_need_an_explanatory_flag(tmp_path):
    with pytest.raises(IngestError, match="created a row without an explanatory flag"):
        _run(tmp_path, {**KEEP, "name": "synthetic.add_row", "params": {"flags": []}})
    res = _run(tmp_path / "ok", {**KEEP, "name": "synthetic.add_row", "params": {"flags": ["lattice_reading"]}})
    assert len(_obs(res, "patch_a", 1)) == 5


def test_unknown_transform_or_plugin_fails(tmp_path):
    root = syn.write_wide_source(tmp_path / "raw")
    with pytest.raises(TransformError, match="Unknown transform"):
        ingest_source(_with_transform({"name": "synthetic.nope", "basis": "structural"}), root, syn.concepts())
    cfg = syn.wide_config(plugins=["no_such_plugin_module"])
    with pytest.raises(TransformError, match="Cannot import"):
        ingest_source(SourceConfig.from_mapping(cfg), root, syn.concepts())
