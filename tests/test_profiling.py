"""M2b descriptive profiling: metric definitions, missing data, gaps, unknown intervals. Synthetic data only."""

import json

import numpy as np
import pandas as pd
import pytest

from pghd_ready.cli import main as cli_main
from pghd_ready.core import profiling as pr
from pghd_ready.core.ingest import ingest_and_write

from conftest import REPO_ROOT

T0 = pd.Timestamp("2024-01-01 00:00:00")


def _obs(pid, stream, offsets_s, values=None, interp=None, file="p.csv"):
    n = len(offsets_s)
    ts = [pd.NaT if o is None else T0 + pd.Timedelta(seconds=o) for o in offsets_s]
    return pd.DataFrame({
        "participant_id": pid, "source_dataset": "syn", "stream_id": stream, "concept": "c",
        "timestamp": pd.to_datetime(ts), "value": values if values is not None else np.arange(n, dtype=float),
        "unit": "u", "raw_value": "1", "is_interpolated": pd.array(interp if interp is not None else [pd.NA] * n,
                                                                    dtype="boolean"),
        "quality_flags": [[] for _ in range(n)], "source_file": file, "source_row": pd.array(range(2, n + 2), dtype="Int64"),
        "source_column": "x"})


def _streams(**intervals):
    return pd.DataFrame([{"participant_id": "syn:1", "stream_id": s, "declared_native_interval_s": i}
                         for s, i in intervals.items()])


def _row(profile, pid="syn:1", stream="cgm"):
    p = profile.stream_profile
    return p[(p.participant_id == pid) & (p.stream_id == stream)].iloc[0]


# ------------------------------------------------------------------ metric definitions
def test_counts_span_days_and_intervals():
    obs = _obs("syn:1", "cgm", [0, 300, 600, 600, 900, 86400 + 300])          # one duplicate timestamp
    r = _row(pr.profile_observations(obs, _streams(cgm=None)))
    assert (r.n_observations, r.n_distinct_timestamps, r.n_duplicate_timestamp_rows) == (6, 5, 1)
    assert r.observed_span_hours == pytest.approx((86400 + 300) / 3600)
    assert (r.calendar_days_spanned, r.days_with_observations) == (2, 2)
    # intervals over distinct timestamps: 300, 300, 300, 85800
    assert (r.n_intervals, r.interval_min_s, r.interval_median_s, r.interval_max_s) == (4, 300, 300, 85800)
    assert (r.interval_mode_s, r.interval_mode_share) == (300, 0.75)


def test_single_observation_has_no_intervals():
    r = _row(pr.profile_observations(_obs("syn:1", "cgm", [0]), _streams(cgm=300.0)))
    assert r.n_distinct_timestamps == 1 and pd.isna(r.interval_median_s)
    assert r.n_gaps == 0 and r.n_slots == 1 and r.slot_coverage == 1


# ------------------------------------------------------------------ missing data
def test_missing_values_and_timestamps_are_counted_not_dropped_silently():
    obs = _obs("syn:1", "cgm", [0, 300, None, 600], values=[1.0, np.nan, 3.0, 4.0])
    r = _row(pr.profile_observations(obs, _streams(cgm=300.0)))
    assert (r.n_observations, r.n_value_missing, r.n_timestamp_missing) == (4, 1, 1)
    assert r.n_distinct_timestamps == 3          # the row without a timestamp is excluded from temporal metrics


def test_value_present_fraction_uses_source_file_rows():
    obs = _obs("syn:1", "cgm", [0, 60, 120], values=[1.0, np.nan, 3.0], file="a.csv")
    fs = pd.DataFrame({"source_file": ["a.csv"], "n_records": [10]})
    r = _row(pr.profile_observations(obs, _streams(cgm=None), file_stats=fs))
    assert r.source_rows == 10 and r.value_present_fraction == pytest.approx(0.2)
    assert pd.isna(_row(pr.profile_observations(obs, _streams(cgm=None))).value_present_fraction)


# ------------------------------------------------------------------ gaps and slot coverage
def test_gap_definition_boundary_and_totals():
    # E = 300, factor 1.5: 450 s is on time (not a gap), 451 s and 1800 s are gaps
    obs = _obs("syn:1", "cgm", [0, 300, 750, 1201, 3001])
    p = pr.profile_observations(obs, _streams(cgm=300.0))
    r = _row(p)
    assert r.n_gaps == 2
    assert r.total_gap_hours == pytest.approx((451 + 1800) / 3600)
    assert r.longest_gap_hours == pytest.approx(0.5)
    assert sorted(p.gaps["gap_s"]) == [451, 1800]
    assert (p.gaps["gap_end"] - p.gaps["gap_start"]).dt.total_seconds().tolist() == p.gaps["gap_s"].tolist()
    # a stricter factor is an explicit, recorded choice
    p1 = pr.profile_observations(obs, _streams(cgm=300.0), gap_factor=1.0)
    assert _row(p1).n_gaps == 3 and p1.parameters["gap_factor"] == 1.0   # 450, 451, 1800 > 300
    with pytest.raises(ValueError, match="gap_factor"):
        pr.profile_observations(obs, _streams(cgm=300.0), gap_factor=0.5)


def test_slot_coverage_is_not_inflated_by_denser_readings():
    # one-minute readings for 1 h against a documented 15-min interval: 5 slots ([0,900),...,[3600,4500)), all observed
    dense = _obs("syn:1", "cgm", list(range(0, 3601, 60)))
    r = _row(pr.profile_observations(dense, _streams(cgm=900.0)))
    assert (r.n_slots, r.n_slots_observed, r.slot_coverage, r.n_gaps) == (5, 5, 1.0, 0)
    # a 2-h hole in the same 1-min grid empties slots and is one gap
    holed = _obs("syn:1", "cgm", list(range(0, 3601, 60)) + list(range(3600 + 7200, 3600 + 7200 + 901, 60)))
    r = _row(pr.profile_observations(holed, _streams(cgm=900.0)))
    assert r.n_gaps == 1 and r.longest_gap_hours == pytest.approx(2.0)
    assert r.n_slots == 14 and r.n_slots_observed == 7 and r.slot_coverage == pytest.approx(0.5)


def test_daily_slot_coverage_of_calendar_day():
    obs = _obs("syn:1", "cgm", [0, 900, 1800, 86400])
    d = pr.profile_observations(obs, _streams(cgm=900.0)).daily_profile.sort_values("date")
    assert d["n_observations"].tolist() == [3, 1] and d["slots_per_day"].tolist() == [96, 96]
    assert d["slot_coverage_of_day"].tolist() == pytest.approx([3 / 96, 1 / 96])


# ------------------------------------------------------------------ unknown / configured intervals
def test_unknown_interval_means_no_gap_or_slot_metrics():
    obs = _obs("syn:1", "hr", [0, 60, 7200])
    p = pr.profile_observations(obs, _streams(hr=None))
    r = _row(p, stream="hr")
    assert r.expected_interval_basis == "none" and pd.isna(r.expected_interval_s)
    assert pd.isna(r.n_gaps) and pd.isna(r.slot_coverage) and p.gaps.empty
    assert r.interval_median_s == pytest.approx(3600)  # observed intervals are still described
    s = p.dataset_summary.set_index("stream_id").loc["hr"]
    assert pd.isna(s.get("n_gaps_total", np.nan))
    assert p.parameters["expected_intervals"]["hr"] == {"seconds": None, "basis": "none"}


def test_configured_interval_is_used_and_labelled_and_never_overrides_documented():
    obs = _obs("syn:1", "hr", [0, 60, 7200])
    r = _row(pr.profile_observations(obs, _streams(hr=None), configured_intervals={"hr": 60}), stream="hr")
    assert (r.expected_interval_s, r.expected_interval_basis, r.n_gaps) == (60, "configured", 1)
    with pytest.raises(ValueError, match="contradicts the documented"):
        pr.profile_observations(obs, _streams(hr=300.0), configured_intervals={"hr": 60})
    for bad in (0, -5, float("nan")):
        with pytest.raises(ValueError, match="positive"):
            pr.profile_observations(obs, _streams(hr=None), configured_intervals={"hr": bad})


# ------------------------------------------------------------------ sampling origin
def test_origin_reported_by_evidence_and_never_inferred_from_regular_timestamps():
    regular_unknown = _obs("syn:1", "cgm", list(range(0, 3000, 300)))                    # perfectly regular, origin NA
    declared = _obs("syn:1", "dev", [0, 60, 120], interp=[False, False, True])
    log = [{"operation": "sampling_origin", "target": "dev", "basis": "documented"}]
    p = pr.profile_observations(pd.concat([regular_unknown, declared], ignore_index=True),
                                _streams(cgm=300.0, dev=None), log=log)
    r = _row(p)
    assert r.origin_unknown == 10 and r.origin_native_declared == 0 and r.origin_native_inferred == 0
    d = _row(p, stream="dev")
    assert (d.origin_native_declared, d.origin_interpolated_declared, d.origin_unknown) == (2, 1, 0)
    # without a declaration in the log the origin is reported as unattributed, not as declared
    nolog = _row(pr.profile_observations(declared, _streams(dev=None)), stream="dev")
    assert (nolog.origin_native_unattributed, nolog.origin_native_declared) == (2, 0)


# ------------------------------------------------------------------ read-only, aggregation
def test_inputs_unchanged_and_summary_aggregates_participants():
    obs = pd.concat([_obs("syn:1", "cgm", [0, 300, 1200]), _obs("syn:2", "cgm", [0, 300])], ignore_index=True)
    before = obs.copy(deep=True)
    p = pr.profile_observations(obs, _streams(cgm=300.0))
    pd.testing.assert_frame_equal(obs, before)
    s = p.dataset_summary.set_index("stream_id").loc["cgm"]
    assert (s.n_participants, s.n_observations, s.n_gaps_total, s.participants_with_gaps) == (2, 5, 1, 1)
    assert "participant_id" not in p.dataset_summary.columns


# ------------------------------------------------------------------ end to end on the synthetic demo source
def test_profile_m1_run_on_synthetic_demo(tmp_path):
    demo = REPO_ROOT / "data" / "synthetic" / "wearable_demo"
    m1 = tmp_path / "m1"
    ingest_and_write(demo / "config" / "source.yaml",
                     [REPO_ROOT / "configs/concepts/shared.yaml", REPO_ROOT / "configs/conditions/cardiometabolic.yaml"],
                     demo / "raw", m1, base_dir=tmp_path)
    m1_bytes = {p.name: p.read_bytes() for p in m1.glob("*.parquet")}
    out = tmp_path / "quality"
    assert cli_main(["profile", "--m1", str(m1), "--out", str(out), "--base-dir", str(tmp_path)]) == 0
    assert {p.name: p.read_bytes() for p in m1.glob("*.parquet")} == m1_bytes          # M1 untouched
    for name in ("stream_profile", "daily_profile", "gaps", "dataset_summary"):
        assert (out / f"{name}.parquet").exists()
    manifest = json.loads((out / "profile_manifest.json").read_text())
    assert {i["path"] for i in manifest["inputs"]} >= {"m1/observations.parquet", "m1/_provenance/manifest.json"}
    assert "gap_factor" in manifest["source"]["parameters"]
    report = (out / "profile_report.md").read_text()
    assert "not device wear-time" in report.replace("**", "")
    obs = pd.read_parquet(m1 / "observations.parquet")
    assert not any(pid in report for pid in obs["participant_id"].unique())   # aggregate only
    # reruns are byte-identical for the data tables
    out2 = tmp_path / "quality2"
    assert cli_main(["profile", "--m1", str(m1), "--out", str(out2), "--base-dir", str(tmp_path)]) == 0
    for name in ("stream_profile", "daily_profile", "gaps", "dataset_summary"):
        assert (out / f"{name}.parquet").read_bytes() == (out2 / f"{name}.parquet").read_bytes()


def test_cli_rejects_malformed_expected_interval(capsys):
    with pytest.raises(SystemExit):
        cli_main(["profile", "--m1", "x", "--out", "y", "--expected-interval", "fitbit_hr"])
