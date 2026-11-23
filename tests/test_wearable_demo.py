"""The persisted demo dataset through the real CLI: outputs, report accuracy, reproducibility, failure report."""

import json
import re
import shutil

import pandas as pd
import pytest

from pghd_ready import cli
from pghd_ready.core.ingest import OUTPUT_FILES
from pghd_ready.core.provenance import sha256_file

import synthetic_sources as syn

CONCEPTS = [str(p) for p in syn.CONCEPT_FILES]
ARGS = ["ingest", "--source", str(syn.DEMO_CONFIG), "--concepts", *CONCEPTS,
        "--data-root", str(syn.DEMO_DIR / "raw")]


def _run(out, base):
    return cli.main(ARGS + ["--out", str(out), "--base-dir", str(base)])


@pytest.fixture(scope="module")
def demo_run(tmp_path_factory):
    before = {p: sha256_file(p) for p in sorted((syn.DEMO_DIR / "raw").rglob("*.csv"))}
    out = tmp_path_factory.mktemp("demo") / "out"
    assert _run(out, syn.REPO_ROOT) == 0
    after = {p: sha256_file(p) for p in before}
    assert before == after, "ingestion must never modify source files"
    return out


def test_all_expected_outputs_are_generated(demo_run):
    missing = [f for f in OUTPUT_FILES if not (demo_run / f).is_file()]
    assert not missing


def test_report_status_and_counts_match_the_run(demo_run):
    report = (demo_run / "ingestion_report.md").read_text()
    summary = json.loads((demo_run / "ingest_summary.json").read_text())
    assert "COMPLETED WITH WARNINGS" in report
    assert summary["rows"] == {"observations": 38, "events": 8, "participant_attributes": 19, "streams": 15,
                               "rejected_records": 7}
    for name, n in summary["rows"].items():
        if name != "rejected_records":
            assert re.search(rf"\| {name} \| {n} \|", report), name
    # per-table read statistics: files, physical lines, data records, used, rejected, blank, padded
    assert "| device_export | 3 | 20 | 15 | 11 | 4 | 1 | 1 |" in report
    assert "| app_export | 1 | 9 | 8 | 6 | 2 | 0 | 0 |" in report
    assert "| profile | 1 | 6 | 5 | 4 | 1 | 0 | 0 |" in report
    assert "21 of 28 source records (75.0%) were used" in report
    rej = pd.read_parquet(demo_run / "rejected_records.parquet")
    for reason, n in rej.groupby("reason").size().items():
        assert f"`{reason}`" in report
    assert "A-DEMO-TZ-01" in report and "documented_error_code" in report
    assert "`Glucose`" in report  # unselected filter value surfaced
    assert "not implemented yet" in report and "not** an assessment of data quality" in report


def test_report_shows_sampling_origin_and_domains(demo_run):
    report = (demo_run / "ingestion_report.md").read_text()
    summary = json.loads((demo_run / "ingest_summary.json").read_text())
    # only the app export is declared native; every device stream stays unknown by default
    assert "| app_glucose | 4 | 0 | 0 | 0 | 0 | declared native (documented:" in report
    assert "| patch_a | 0 | 0 | 0 | 0 | 10 | not declared or classified - unknown by default |" in report
    origin = {r["stream_id"]: r for r in summary["sampling_origin_by_stream"]}
    assert sum(r["unknown"] for r in origin.values()) == 34 and origin["app_glucose"]["native"] == 4
    # domain table: tags are descriptive; demographics untagged
    assert "PGHD domains represented" in report
    assert "| behavioural | 9 | 8 | 0 |" in report  # wrist_met 6 + wrist_intensity 3; 8 events
    assert "| (untagged) | 0 | 0 | 8 |" in report  # sex 4 + ldl 4
    assert {d["domain"] for d in summary["records_by_domain"]} == {"physiological", "behavioural", None}


def test_report_does_not_overclaim(demo_run):
    report = (demo_run / "ingestion_report.md").read_text().lower()
    for phrase in ("high quality", "data quality is good", "good quality"):
        assert phrase not in report
    # any statement about suitability/readiness must be a negation
    sentences = re.split(r"(?<=[.!?])\s+|\n", report)
    for s in sentences:
        if "suitable" in s or "ready for" in s:
            assert " not " in f" {s} ", s


def test_run_is_reproducible(demo_run, tmp_path):
    out2 = tmp_path / "out2"
    assert _run(out2, syn.REPO_ROOT) == 0
    for name in ("observations", "events", "participant_attributes", "streams", "rejected_records"):
        pd.testing.assert_frame_equal(pd.read_parquet(demo_run / f"{name}.parquet"),
                                      pd.read_parquet(out2 / f"{name}.parquet"))

    def body(p):  # the report differs only in its generation timestamp and output paths
        txt = (p / "ingestion_report.md").read_text()
        return [l for l in txt.splitlines() if not l.startswith("Generated") and "/out" not in l]

    assert body(demo_run) == body(out2)


def test_failed_run_writes_a_failure_report(tmp_path):
    raw = syn.write_wide_source(tmp_path / "raw")
    (raw / "device_export/user_03/user_03.csv").write_text("Timestamp,Sensor B\n2000-01-01 00:00:00,1\n")
    out = tmp_path / "out"
    (out).mkdir()
    (out / "observations.parquet").write_text("stale output from an earlier run")
    code = cli.main(["ingest", "--source", str(syn.DEMO_CONFIG), "--concepts", *CONCEPTS,
                     "--data-root", str(raw), "--out", str(out), "--base-dir", str(tmp_path)])
    assert code == 1
    report = (out / "ingestion_report.md").read_text()
    assert "FAILED" in report and "required column for observation:patch_a" in report
    assert "No tables or audit files were written" in report
    assert not (out / "observations.parquet").exists()  # stale outputs are removed, never mixed with a failure


def test_clean_source_reports_success(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "hr.csv").write_text("t,id,bpm\n2000-01-01T00:00:00Z,1,60\n2000-01-01T00:01:00Z,1,61\n")
    cfg = {"source": "clean", "version": "1", "temporal": {"output_timezone": "UTC"},
           "tables": [{"name": "hr", "files": {"path": "hr.csv"}, "participant_id": {"from": "column", "column": "id"},
                       "timestamp": {"column": "t", "format": "ISO8601"},
                       "observations": [{"stream_id": "hr", "concept": "heart_rate", "column": "bpm",
                                         "unit": "/min"}]}]}
    src = syn.write_config(tmp_path / "clean.yaml", cfg)
    out = tmp_path / "out"
    assert cli.main(["ingest", "--source", str(src), "--concepts", *CONCEPTS, "--data-root",
                     str(raw), "--out", str(out), "--base-dir", str(tmp_path)]) == 0
    report = (out / "ingestion_report.md").read_text()
    assert "SUCCESS" in report and "Every source record was used" in report
    assert "not** an assessment of data quality" in report


def test_demo_dataset_contains_no_real_data_markers():
    text = " ".join(p.read_text(encoding="utf-8-sig") for p in (syn.DEMO_DIR / "raw").rglob("*.csv"))
    for marker in ("CGMacros", "Libre", "Dexcom", "Fitbit"):
        assert marker.lower() not in text.lower()
    assert sum(1 for _ in (syn.DEMO_DIR / "raw").rglob("*.csv")) == 5
    assert all(p.stat().st_size < 2_000 for p in (syn.DEMO_DIR / "raw").rglob("*.csv"))


def test_copy_of_demo_is_independent(tmp_path):
    root = syn.write_wide_source(tmp_path / "raw")
    (root / "profile.csv").write_text("changed")
    assert (syn.DEMO_DIR / "raw" / "profile.csv").read_text() != "changed"
    shutil.rmtree(root)
