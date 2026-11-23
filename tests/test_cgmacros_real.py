"""CGMacros base ingestion (M1c-1): reconciliation against an independent reference profiler.

Runs only where the CGMacros data are present locally (skipped otherwise). Assertions compare
aggregates; failure messages report counts of mismatches, never participant-level values.
"""

import csv
import json
import random
from collections import Counter
from functools import lru_cache

import numpy as np
import pandas as pd
import pytest

import cgmacros_reference as ref
from pghd_ready.core.concepts import ConceptRegistry
from pghd_ready.core.config import SourceConfig
from pghd_ready.core.ingest import ingest_source, write_outputs

pytestmark = [pytest.mark.real_data,
              pytest.mark.skipif(not ref.available(), reason="CGMacros data not available locally")]

CONFIG = ref.REPO_ROOT / "configs" / "sources" / "cgmacros.yaml"
CONCEPTS = [ref.REPO_ROOT / "configs" / "concepts" / "shared.yaml",
            ref.REPO_ROOT / "configs" / "conditions" / "cardiometabolic.yaml"]
MAPPED_LABELS = {"breakfast", "lunch", "dinner", "snack", "snacks"}


@pytest.fixture(scope="module")
def run():
    return ingest_source(SourceConfig.from_yaml(CONFIG), ref.CGM_ROOT, ConceptRegistry.from_yaml_files(CONCEPTS))


@pytest.fixture(scope="module")
def reference():
    return ref.profile()


def _raw_sum(series: pd.Series) -> float:
    return float(pd.to_numeric(series, errors="coerce").sum())


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= 1e-6 * max(1.0, abs(a), abs(b))


# ---------------------------------------------------------------- records and participants
def test_every_source_record_is_accounted_for(run, reference):
    fs = run.file_stats.set_index("source_file")
    ts_files = {f: r for f, r in reference["files"].items()}
    bad = sum(int(fs.loc[f, "n_records"] != r["rows"]) for f, r in ts_files.items())
    assert bad == 0, f"{bad} file(s) with a record count different from the reference"
    assert int(fs.loc[list(ts_files), "n_rejected"].sum()) == sum(r["empty_timestamps"] for r in ts_files.values())
    assert int(fs.loc["bio.csv", "n_records"]) == reference["bio_participants"]
    assert len(run.rejected_records) == 0
    assert int(fs["n_blank_lines"].sum()) == 0 and int(fs["n_short_rows_padded"].sum()) == 0


def test_participants(run, reference):
    obs_p = set(run.observations["participant_id"].unique())
    att_p = set(run.participant_attributes["participant_id"].unique())
    assert len(obs_p) == len(reference["files"]) == 45
    assert obs_p == att_p and len(att_p) == reference["bio_participants"]


# ---------------------------------------------------------------- observations
def test_stream_counts_and_raw_sums_match_reference(run, reference):
    st = run.streams.set_index(["participant_id", "stream_id"])
    obs = run.observations
    sums = obs.groupby(["participant_id", "stream_id"])["raw_value"].apply(_raw_sum)
    assert set(st.index) == set(reference["streams"]), "stream set differs from the reference"
    bad_n = sum(int(st.loc[k, "n_rows_source"] != r["n"]) for k, r in reference["streams"].items())
    bad_sum = sum(int(not _close(sums.loc[k], r["sum"])) for k, r in reference["streams"].items())
    assert bad_n == 0, f"{bad_n} participant-stream pair(s) with differing counts"
    assert bad_sum == 0, f"{bad_sum} participant-stream pair(s) with differing raw sums"
    assert (st["n_rows_source"] == st["n_rows_output"]).all()  # no transforms in M1c-1


def test_values_equal_raw_except_documented_scale(run):
    obs = run.observations
    raw = pd.to_numeric(obs["raw_value"], errors="coerce")
    met = obs["stream_id"] == "fitbit_met"
    assert np.allclose(obs.loc[met, "value"], raw[met] * 0.1, rtol=0, atol=1e-9)
    assert (obs.loc[~met, "value"] == raw[~met]).all()
    assert obs["value"].notna().all()  # every non-empty source cell parsed


def test_cgm_limit_and_range_flags_match_reference(run, reference):
    obs = run.observations
    for stream in ("libre_pro", "dexcom_g6"):
        s = obs[obs["stream_id"] == stream]["quality_flags"]
        at_limit = sum("at_documented_reporting_limit" in f for f in s)
        outside = sum("outside_documented_range" in f for f in s)
        ref_rows = [r for (p, st), r in reference["streams"].items() if st == stream]
        assert at_limit == sum(r["at_limit"] for r in ref_rows), stream
        assert outside == sum(r["outside"] for r in ref_rows), stream
    for stream in ("fitbit_hr", "fitbit_activity_kcal", "fitbit_met"):
        s = obs[obs["stream_id"] == stream]["quality_flags"]
        ref_out = sum(r["outside"] for (p, st), r in reference["streams"].items() if st == stream)
        assert sum("outside_documented_range" in f for f in s) == ref_out, stream


def test_sampling_origin_is_unknown_everywhere(run):
    assert run.observations["is_interpolated"].isna().all()
    st = run.streams
    assert (st["n_sampling_unknown"] == st["n_rows_output"]).all()
    assert not any(r.operation in ("transform", "sampling_origin") for r in run.transform_log.records)


def test_timestamps_within_reference_bounds(run, reference):
    obs = run.observations.groupby("participant_id")["timestamp"].agg(["min", "max"])
    outside = sum(int(obs.loc[p, "min"] < lo or obs.loc[p, "max"] > hi) for p, (lo, hi) in reference["ts_bounds"].items())
    exact = sum(int(obs.loc[p, "min"] == lo and obs.loc[p, "max"] == hi) for p, (lo, hi) in reference["ts_bounds"].items())
    assert outside == 0
    assert exact == len(reference["ts_bounds"]), f"{len(reference['ts_bounds']) - exact} participant(s) whose first/last record time differs"


@lru_cache(maxsize=None)
def _lines(rel: str) -> tuple[list[str], list[str]]:
    with open(ref.CGM_ROOT / rel, encoding="utf-8-sig", newline="") as fh:
        text = fh.read().splitlines()
    return next(csv.reader([text[0]])), text


def test_raw_values_found_at_source_line(run):
    """Random spot check: raw_value equals the CSV field at the recorded physical line and column."""
    rng = random.Random(20261123)
    obs = run.observations
    sample = obs.iloc[rng.sample(range(len(obs)), 500)]
    bad = 0
    for r in sample.itertuples():
        header, lines = _lines(r.source_file)
        fields = next(csv.reader([lines[int(r.source_row) - 1]]))
        bad += fields[header.index(r.source_column)] != r.raw_value
    assert bad == 0, f"{bad} of 500 sampled observations do not match their source line"


# ---------------------------------------------------------------- events
def test_meal_and_photo_events_match_reference(run, reference):
    ev = run.events
    meals = ev[ev["event_type"] == "meal"]
    photos = ev[ev["event_type"] == "meal_photo"]
    m_n, p_n = meals.groupby("participant_id").size(), photos.groupby("participant_id").size()
    bad = sum(int(m_n.get(p, 0) != r["n"] or p_n.get(p, 0) != r["photos"]) for p, r in reference["meals"].items())
    assert bad == 0, f"{bad} participant(s) with differing meal/photo counts"
    ref_labels = sum((r["labels"] for r in reference["meals"].values()), Counter())
    assert Counter(meals["raw_label"]) == ref_labels
    unmapped = sum(n for lab, n in ref_labels.items() if lab.strip().lower() not in MAPPED_LABELS)
    assert sum("unmapped_category" in f for f in meals["quality_flags"]) == unmapped
    assert meals.loc[meals["event_subtype"].isna(), "raw_label"].map(lambda s: s.strip().lower()).isin(
        MAPPED_LABELS).sum() == 0


def test_meal_attributes_preserve_raw_and_match_reference(run, reference):
    meals = run.events[run.events["event_type"] == "meal"]
    raw = pd.DataFrame([json.loads(x) for x in meals["raw_attributes"]], index=meals.index)
    for target in ("energy_kcal", "carbohydrate_g", "protein_g", "fat_g", "fiber_g", "amount_consumed_raw"):
        ref_rows = [r[target] for r in reference["meals"].values() if target in r]
        assert int(raw[target].notna().sum()) == sum(r["n"] for r in ref_rows), target
        assert _close(_raw_sum(raw[target]), sum(r["sum"] for r in ref_rows)), target
        assert np.allclose(meals[target].dropna(), pd.to_numeric(raw.loc[meals[target].notna(), target])), target
    flag_out = Counter()
    for flags in meals["quality_flags"]:
        flag_out.update(f for f in flags if "documented" in f)

    def ref(target, key):
        return sum(r[target][key] for r in reference["meals"].values() if target in r)

    # a meal can match rules on several attributes; the flag is per event, so per-attribute matches come from the
    # log's values_matched (flags_raised only counts newly added event flags)
    logged = {(r.target, r.params["flag"]): r.params["values_matched"]
              for r in run.transform_log.records if r.operation == "rule"}
    for t in ("energy_kcal", "carbohydrate_g", "fat_g", "fiber_g"):
        assert logged[(f"meal.{t}", "outside_documented_range")] == ref(t, "outside"), t
    # protein: below the documented minimum is a contextual review flag, above the maximum a range anomaly
    assert logged[("meal.protein_g", "below_documented_minimum_review")] == ref("protein_g", "below")
    assert logged[("meal.protein_g", "outside_documented_range")] == ref("protein_g", "above")
    assert flag_out["below_documented_minimum_review"] == ref("protein_g", "below")
    low_protein = meals[meals["quality_flags"].map(lambda f: "below_documented_minimum_review" in f)]
    assert low_protein["protein_g"].notna().all()  # values preserved, not removed or emptied
    assert logged[("meal.amount_consumed_raw", "amount_outside_documented_range")] == ref("amount_consumed_raw", "outside") \
        == flag_out["amount_outside_documented_range"]


# ---------------------------------------------------------------- participant attributes
def test_bio_attributes_match_reference_and_preserve_raw(run, reference):
    att = run.participant_attributes
    for concept, r in reference["attributes"].items():
        sub = att[att["concept"] == concept]
        assert len(sub) == r["n"], concept
        if "sum" in r:
            assert _close(_raw_sum(sub["raw_value"]), r["sum"]), concept
        if "labels" in r:
            assert Counter(sub["value_text"]) == r["labels"], concept  # labels preserved, not harmonised


def test_bio_range_flags_use_documented_precision(run, reference):
    """Range flags on bio attributes match the reference (both use half-last-digit tolerance); values unrounded."""
    att = run.participant_attributes
    flagged = att[att["quality_flags"].map(lambda f: "outside_documented_range" in f)].groupby("concept").size()
    for concept, r in reference["attributes"].items():
        if r.get("outside") is not None and concept not in ("ldl_cholesterol_calculated", "vldl_cholesterol_calculated"):
            assert int(flagged.get(concept, 0)) == r["outside"], concept
    bmi = att[att["concept"] == "body_mass_index"]
    assert int(flagged.get("body_mass_index", 0)) == 0  # the rounding artefact at the documented minimum is gone
    assert (bmi["value_num"] == pd.to_numeric(bmi["raw_value"])).all()  # no rounding of stored values


def test_documented_errors_and_anomalies_in_bio(run, reference):
    att = run.participant_attributes
    for concept, code in (("ldl_cholesterol_calculated", 800.0), ("vldl_cholesterol_calculated", 400.0)):
        sub = att[att["concept"] == concept]
        hit = pd.to_numeric(sub["raw_value"]) == code
        assert hit.sum() == reference["attributes"][concept]["values"].count(code) == 1
        assert sub.loc[hit, "value_num"].isna().all()
        assert all("documented_error_code" in f for f in sub.loc[hit, "quality_flags"])
        assert sub.loc[~hit, "value_num"].notna().all()
    ratio = att[att["concept"] == "cholesterol_hdl_ratio"]
    hit = pd.to_numeric(ratio["raw_value"]) == 400.0
    assert hit.sum() == 1 and ratio.loc[hit, "value_num"].notna().all()  # flagged, not emptied
    assert all("suspected_implausible_value" in f for f in ratio.loc[hit, "quality_flags"])


def test_hba1c_kept_as_reported_and_flagged(run):
    a = run.participant_attributes[run.participant_attributes["concept"] == "hba1c"]
    assert len(a) == 45 and (a["unit"] == "%").all()
    assert (a["value_num"] == pd.to_numeric(a["raw_value"])).all()  # numeric value unchanged, no conversion
    assert all({"hba1c_documented_unit_conflict", "hba1c_unit_assumed_percent"} <= set(f) for f in a["quality_flags"])
    assert a["measured_at"].isna().all() and a["measured_at_raw"].notna().all()


def test_unit_conversions_and_no_invented_dates(run):
    att = run.participant_attributes
    w = att[att["concept"] == "body_weight"]
    assert np.allclose(w["value_num"], pd.to_numeric(w["raw_value"]) * 0.45359237)
    fs = att[att["concept"] == "capillary_glucose"]
    assert len(fs) == 135 and fs["measured_at"].isna().all() and fs["measured_at_raw"].notna().all()
    assert run.config.temporal.output_timezone is None and run.config.temporal.date_shifted is True


# ---------------------------------------------------------------- outputs
def test_outputs_written_and_summary_is_aggregate(run, tmp_path):
    out = tmp_path / "cgmacros"
    write_outputs(run, out, base_dir=ref.REPO_ROOT, record_git=False)
    summary = (out / "ingest_summary.json").read_text()
    assert "cgmacros:0" not in summary  # no participant identifiers in the machine-readable summary
    report = (out / "ingestion_report.md").read_text()
    assert "COMPLETED WITH WARNINGS" in report and "A-CGMACROS-BIO-01" in report
    assert "not declared or classified - unknown by default" in report
