"""Three-state sampling origin: representation, declarations, classification transforms and inclusion policies."""

import json

import pandas as pd
import pytest

from pghd_ready.core.config import ConfigError, SourceConfig
from pghd_ready.core.ingest import IngestError, ingest_source, write_outputs
from pghd_ready.core.sampling import InclusionPolicy, origin_counts, origin_labels, select_observations

import synthetic_sources as syn
from synthetic_sources import APP, DEVICE, PID
from conftest import make_observations


# ---------------------------------------------------------------- pure representation
def test_origin_labels_and_policies():
    obs = make_observations()  # is_interpolated = False, <NA>, True
    assert origin_labels(obs["is_interpolated"]).tolist() == ["native", "unknown", "interpolated"]
    assert select_observations(obs, "native_only").index.tolist() == [0]
    assert select_observations(obs, InclusionPolicy.NATIVE_AND_UNKNOWN).index.tolist() == [0, 1]
    assert select_observations(obs, "all").index.tolist() == [0, 1, 2]
    with pytest.raises(ValueError):
        select_observations(obs, "everything_please")


def test_origin_counts_always_has_three_columns():
    obs = make_observations()
    obs["is_interpolated"] = pd.array([pd.NA] * 3, dtype="boolean")
    c = origin_counts(obs).iloc[0]
    assert (c["native"], c["interpolated"], c["unknown"]) == (0, 0, 3)


# ---------------------------------------------------------------- ingestion defaults and declarations
@pytest.fixture
def demo(tmp_path):
    root = syn.write_wide_source(tmp_path / "raw")
    return root, ingest_source(SourceConfig.from_mapping(syn.wide_config()), root, syn.concepts())


def test_default_origin_is_unknown(demo):
    _, res = demo
    dev = res.observations[res.observations.stream_id == "patch_a"]
    assert dev.is_interpolated.isna().all()
    st = res.streams.set_index(["participant_id", "stream_id"]).loc[(PID.format(1), "patch_a")]
    assert (st.n_sampling_native, st.n_sampling_interpolated, st.n_sampling_unknown) == (0, 0, 4)


def test_declared_origin_requires_basis_and_is_logged(demo):
    _, res = demo
    app = res.observations[res.observations.stream_id == "app_glucose"]
    assert (app.is_interpolated == False).all()  # noqa: E712  declared native (documented)
    step = next(r for r in res.transform_log.records if r.operation == "sampling_origin")
    assert step.basis.value == "documented" and step.evidence and step.params["declared_origin"] == "native"
    cfg = syn.wide_config()
    cfg["tables"][APP]["observations"][0]["sampling_origin"] = {"value": "native", "basis": "structural"}
    with pytest.raises(ConfigError, match="sampling-origin declaration needs basis"):
        SourceConfig.from_mapping(cfg)
    cfg["tables"][APP]["observations"][0]["sampling_origin"] = {"value": "probably", "basis": "documented",
                                                                 "evidence": "x"}
    with pytest.raises(ConfigError, match="'native' or 'interpolated'"):
        SourceConfig.from_mapping(cfg)


# ---------------------------------------------------------------- inferred classification via a transform
CLASSIFY = {"name": "synthetic.classify_origin", "params": {"lattice_min": 2, "min_readings": 4},
            "basis": "assumption", "assumption_ids": ["A-SYN-01"],
            "flags": ["inferred_native", "inferred_interpolated", "origin_undetermined"]}


def _classified(tmp_path, transform=CLASSIFY):
    cfg = syn.wide_config(plugins=["synthetic_plugin"])
    cfg["tables"][DEVICE]["observations"][0]["transforms"] = [transform]
    root = syn.write_wide_source(tmp_path / "classified_raw")
    return ingest_source(SourceConfig.from_mapping(cfg), root, syn.concepts())


def test_classification_preserves_values_and_records_evidence(tmp_path, demo):
    _, plain = demo
    res = _classified(tmp_path)
    a = res.observations[res.observations.stream_id == "patch_a"].sort_values(["participant_id", "source_row"])
    b = plain.observations[plain.observations.stream_id == "patch_a"].sort_values(["participant_id", "source_row"])
    # original values, raw values and timestamps are untouched; nothing removed or rejected
    for col in ("value", "raw_value", "timestamp", "source_row"):
        pd.testing.assert_series_equal(a[col].reset_index(drop=True), b[col].reset_index(drop=True))
    assert not (res.rejected_records.reason == "removed_by_transform").any()
    p1 = a[a.participant_id == PID.format(1)]  # 4 readings -> enough evidence; minutes 0,1,2,3 UTC
    assert origin_labels(p1.is_interpolated).tolist() == ["native", "interpolated", "native", "interpolated"]
    assert [f[-1] for f in p1.quality_flags] == ["inferred_native", "inferred_interpolated"] * 2
    p2 = a[a.participant_id == PID.format(2)]  # 3 readings -> insufficient -> unknown
    assert p2.is_interpolated.isna().all() and all("origin_undetermined" in f for f in p2.quality_flags)
    # method, evidence and confidence recorded per participant x stream
    d = res.stream_diagnostics
    p1d = dict(zip(d[d.participant_id == PID.format(1)].key, d[d.participant_id == PID.format(1)].value))
    assert p1d == {"method": "toy_minute_lattice_v0", "lattice_min": "2", "evidence_readings": "4", "confidence": "0.9"}
    rec = next(r for r in res.transform_log.records if r.operation == "transform")
    assert rec.assumption_ids == ["A-SYN-01"] and rec.rows_in == rec.rows_out


def test_streams_and_summary_report_three_counts(tmp_path):
    res = _classified(tmp_path)
    st = res.streams.set_index(["participant_id", "stream_id"])
    s1, s2 = st.loc[(PID.format(1), "patch_a")], st.loc[(PID.format(2), "patch_a")]
    assert (s1.n_sampling_native, s1.n_sampling_interpolated, s1.n_sampling_unknown) == (2, 2, 0)
    assert (s2.n_sampling_native, s2.n_sampling_interpolated, s2.n_sampling_unknown) == (0, 0, 3)
    by_stream = {r["stream_id"]: r for r in res.summary()["sampling_origin_by_stream"]}
    assert by_stream["patch_a"] == {"stream_id": "patch_a", "native": 2, "interpolated": 2, "unknown": 6}
    assert by_stream["app_glucose"]["native"] == 4


def test_sensitivity_policies_on_classified_stream(tmp_path):
    res = _classified(tmp_path)
    a = res.observations[res.observations.stream_id == "patch_a"]
    sizes = {p.value: len(select_observations(a, p)) for p in InclusionPolicy}
    assert sizes == {"native_only": 2, "native_and_unknown": 8, "all": 10}


def test_report_shows_three_classifications_and_diagnostics(tmp_path):
    res = _classified(tmp_path)
    out = tmp_path / "out"
    write_outputs(res, out, base_dir=tmp_path, record_git=False)
    report = (out / "ingestion_report.md").read_text()
    assert "| patch_a | 0 | 2 | 0 | 2 | 6 | classified by `synthetic.classify_origin` (assumption: A-SYN-01) |" in report
    assert "| app_glucose | 4 | 0 | 0 | 0 | 0 | declared native (documented:" in report
    assert "not declared or classified - unknown by default" in report
    assert "| `synthetic.classify_origin` | patch_a | confidence | 3 | min 0, median 0, max 0.9 |" in report
    assert "native only, native and unknown, or all" in report
    assert (out / "_provenance" / "stream_diagnostics.csv").exists()


def test_declared_and_inferred_native_are_distinguishable_from_written_outputs(tmp_path):
    """Both are admitted by native_only, but downstream research can separate them from the outputs alone."""
    from pghd_ready.core.sampling import origin_evidence, origin_evidence_counts

    cfg = syn.wide_config(plugins=["synthetic_plugin"])
    patch_a = cfg["tables"][DEVICE]["observations"][0]
    # one stream that is declared native (assumption) AND classified: only participant 1 has enough evidence
    patch_a["sampling_origin"] = {"value": "native", "basis": "assumption", "assumption_ids": ["A-SYN-02"]}
    patch_a["transforms"] = [CLASSIFY]
    res = ingest_source(SourceConfig.from_mapping(cfg), syn.write_wide_source(tmp_path / "raw"), syn.concepts())
    out = tmp_path / "out"
    write_outputs(res, out, base_dir=tmp_path, record_git=False)

    # a downstream step reads only the written files
    obs = pd.read_parquet(out / "observations.parquet")
    log = [json.loads(l) for l in (out / "_provenance" / "transform_log.jsonl").read_text().splitlines()]
    obs["origin"] = origin_labels(obs["is_interpolated"])
    obs["evidence"] = origin_evidence(obs, log)

    a = obs[obs.stream_id == "patch_a"]
    native = select_observations(a, "native_only")
    assert len(native) == 8  # 2 inferred (participant 1) + 6 declared (participants 2 and 3)
    assert native.groupby("evidence").size().to_dict() == {"declared": 6, "inferred": 2}
    assert set(native[native.evidence == "inferred"].participant_id) == {PID.format(1)}
    app = obs[obs.stream_id == "app_glucose"]
    assert (app.evidence == "declared").all() and (app.origin == "native").all()
    assert (obs[obs.origin == "unknown"].evidence == "unknown").all()
    # the basis of each declaration is in the same log
    decl = {r["target"]: r["basis"] for r in log if r["operation"] == "sampling_origin"}
    assert decl == {"patch_a": "assumption", "app_glucose": "documented"}

    counts = origin_evidence_counts(obs, log).set_index("stream_id").loc["patch_a"]
    assert counts[["native_declared", "native_inferred", "interpolated_inferred", "unknown"]].tolist() == [6, 2, 2, 0]
    report = (out / "ingestion_report.md").read_text()
    assert ("| patch_a | 6 | 2 | 0 | 2 | 0 | declared native (assumption: A-SYN-02); classified by "
            "`synthetic.classify_origin` (assumption: A-SYN-01) |") in report
    summary = json.loads((out / "ingest_summary.json").read_text())
    ev = {r["stream_id"]: r for r in summary["sampling_origin_evidence_by_stream"]}
    assert ev["patch_a"]["native_declared"] == 6 and ev["patch_a"]["native_inferred"] == 2


def test_classification_without_explanatory_flag_is_rejected(tmp_path):
    with pytest.raises(IngestError, match="changed values without adding an explanatory flag"):
        _classified(tmp_path, {**CLASSIFY, "name": "synthetic.classify_without_flag"})


def test_unknown_survives_parquet_round_trip(tmp_path):
    from pghd_ready.core.io import read_table
    from pghd_ready.core.schemas import observation_schema

    res = _classified(tmp_path)
    out = tmp_path / "out"
    write_outputs(res, out, base_dir=tmp_path, record_git=False)
    back = read_table(out / "observations.parquet", observation_schema("UTC"))
    assert str(back.is_interpolated.dtype) == "boolean"
    pd.testing.assert_series_equal(back.is_interpolated, res.observations.is_interpolated)
