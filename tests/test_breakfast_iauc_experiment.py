"""M4 breakfast iAUC experiment: leakage guards, LOPO, metrics, bootstrap, outputs. Synthetic data only."""

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pghd_ready.conditions.cardiometabolic.experiments import breakfast_iauc as ex
from pghd_ready.core.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "breakfast_iauc.yaml"
DEVICES = ["dev_a", "dev_b"]


def _cfg(**overrides):
    cfg = ex.load_config(CONFIG)
    cfg["devices"] = list(DEVICES)
    cfg["models"]["random_forest"] = {"n_estimators": 20, "min_samples_leaf": 3, "max_features": 1.0, "random_state": 0}
    cfg["bootstrap"]["resamples"] = 200
    cfg.update(overrides)
    return cfg


def _table(n_participants=8, per=6, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_participants):
        effect = rng.normal(0, 500)
        for b in range(per):
            x = {"energy_kcal": rng.uniform(200, 800), "carbohydrate_g": rng.uniform(10, 100),
                 "protein_g": rng.uniform(5, 40), "fat_g": rng.uniform(5, 40), "fiber_g": rng.uniform(0, 10),
                 "baseline_glucose_mgdl": rng.uniform(80, 130), "hba1c": 5 + p * 0.2,
                 "body_mass_index": 22 + p, "age": 30 + p}
            for d_i, d in enumerate(DEVICES):
                y = 40 * x["carbohydrate_g"] - 1500 + effect + rng.normal(0, 300) + d_i * 100
                rows.append({"participant_id": f"P{p:02d}", "event_id": f"P{p:02d}-B{b}", "stream_id": d,
                             "outcome_status": "calculable", "outcome_reason": "calculable",
                             "iauc_signed_mgdl_min": y, "n_outcome_readings": 121,
                             "outcome_max_interval_s": 60.0, **x})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ configuration guards
def test_shipped_config_is_the_approved_design():
    cfg = ex.load_config(CONFIG)
    assert cfg["predictors"] == list(ex.APPROVED_PREDICTORS) and cfg["target"] == "iauc_signed_mgdl_min"
    assert cfg["devices"] == ["libre_pro", "dexcom_g6"]
    assert cfg["validation"] == {"scheme": "leave_one_participant_out", "group": "participant_id"}
    assert cfg["models"]["random_forest"] == {"n_estimators": 500, "min_samples_leaf": 5, "max_features": 1.0,
                                              "random_state": 20261125}
    assert [a["name"] for a in cfg["analyses"]] == ["primary", "no_baseline_glucose", "common_breakfasts"]
    assert cfg["metrics"]["primary"] == "participant_macro_mae"


@pytest.mark.parametrize("mutate", [
    lambda c: c["predictors"].append("n_outcome_readings"),          # outcome-derived column
    lambda c: c["predictors"].append("participant_id"),
    lambda c: c["predictors"].remove("age"),                          # set must stay unchanged
    lambda c: c.update(target="iauc_positive"),
    lambda c: c["validation"].update(scheme="kfold"),
    lambda c: c["analyses"][0].update(drop_predictors=["age"]),        # primary must use every predictor
    lambda c: c["analyses"][1].update(drop_predictors=["n_outcome_readings"]),
    lambda c: c["models"].pop("random_forest"),
])
def test_config_guards(mutate):
    cfg = _cfg()
    mutate(cfg)
    with pytest.raises(ValueError):
        ex.validate_config(cfg)


# ------------------------------------------------------------------ rows and leakage
def test_only_calculable_rows_and_no_imputation():
    t = _table()
    t.loc[0, "outcome_status"] = "not_calculable"
    rows = ex.select_rows(t, DEVICES, "all_calculable")
    assert t.loc[0, "event_id"] not in set(rows[t.loc[0, "stream_id"]]["event_id"])
    t2 = _table()
    t2.loc[3, "fat_g"] = np.nan
    with pytest.raises(ValueError, match="no imputation"):
        ex.select_rows(t2, DEVICES, "all_calculable")


def test_common_breakfasts_are_calculable_on_every_device():
    t = _table()
    t.loc[(t.stream_id == "dev_b") & (t.event_id == "P00-B0"), "outcome_status"] = "not_calculable"
    rows = ex.select_rows(t, DEVICES, "common_breakfasts")
    assert all("P00-B0" not in set(rows[d]["event_id"]) for d in DEVICES)
    assert len(ex.select_rows(t, DEVICES, "all_calculable")["dev_a"]) == len(rows["dev_a"]) + 1


def test_lopo_holds_out_whole_participants():
    cfg, df = _cfg(), ex.select_rows(_table(), DEVICES, "all_calculable")["dev_a"]
    oof = ex.lopo_predictions(df, list(ex.APPROVED_PREDICTORS), cfg, use_forest=False)
    assert oof.groupby("participant_id")["fold"].nunique().eq(1).all()          # all breakfasts together
    assert oof.groupby("fold")["participant_id"].nunique().eq(1).all()          # one participant per fold
    for p, g in oof.groupby("participant_id"):                                   # naive = mean of the others only
        assert np.allclose(g["pred_naive"], df.loc[df.participant_id != p, "iauc_signed_mgdl_min"].mean())
    p = "P03"                                                                    # linear = independent lstsq
    tr = df[df.participant_id != p]
    A = np.column_stack([np.ones(len(tr)), tr[list(ex.APPROVED_PREDICTORS)].to_numpy()])
    beta = np.linalg.lstsq(A, tr["iauc_signed_mgdl_min"].to_numpy(), rcond=None)[0]
    te = df[df.participant_id == p]
    exp = beta[0] + te[list(ex.APPROVED_PREDICTORS)].to_numpy() @ beta[1:]
    assert np.allclose(oof.loc[oof.participant_id == p, "pred_linear"], exp)


def test_test_participant_target_never_reaches_its_own_predictions():
    cfg, t = _cfg(), _table()
    base = ex.lopo_predictions(ex.select_rows(t, DEVICES, "all_calculable")["dev_a"],
                               list(ex.APPROVED_PREDICTORS), cfg, use_forest=False)
    t2 = t.copy()
    t2.loc[t2.participant_id == "P02", "iauc_signed_mgdl_min"] += 1e6
    alt = ex.lopo_predictions(ex.select_rows(t2, DEVICES, "all_calculable")["dev_a"],
                              list(ex.APPROVED_PREDICTORS), cfg, use_forest=False)
    sel = base.participant_id == "P02"
    pd.testing.assert_frame_equal(base.loc[sel, ["pred_naive", "pred_linear"]], alt.loc[sel, ["pred_naive", "pred_linear"]])


def test_outcome_window_columns_do_not_influence_results():
    cfg, t = _cfg(), _table()
    a = ex.run_experiment(cfg, t, use_forest=False)
    t2 = t.assign(n_outcome_readings=np.arange(len(t)), outcome_max_interval_s=np.arange(len(t)) * 7.0)
    b = ex.run_experiment(cfg, t2, use_forest=False)
    assert a.metrics == b.metrics


def test_devices_are_never_pooled():
    cfg, t = _cfg(), _table()
    a = ex.run_experiment(cfg, t, use_forest=False)
    t2 = t.copy()
    t2.loc[t2.stream_id == "dev_b", "iauc_signed_mgdl_min"] *= 3
    b = ex.run_experiment(cfg, t2, use_forest=False)
    assert a.metrics["analyses"]["primary"]["dev_a"] == b.metrics["analyses"]["primary"]["dev_a"]
    assert a.metrics["analyses"]["primary"]["dev_b"] != b.metrics["analyses"]["primary"]["dev_b"]
    assert set(a.metrics["analyses"]["primary"]) == set(DEVICES)               # no pooled entry


def test_signed_target_is_preserved():
    t = _table()
    t.loc[t.index[:5], "iauc_signed_mgdl_min"] = -1000.0
    r = ex.run_experiment(_cfg(), t, use_forest=False)
    prim = r.oof_predictions[r.oof_predictions.analysis == "primary"]
    assert (prim["observed"] < 0).sum() == (t["iauc_signed_mgdl_min"] < 0).sum() >= 5
    assert r.metrics["analyses"]["primary"]["dev_a"]["n_breakfasts"] + \
        r.metrics["analyses"]["primary"]["dev_b"]["n_breakfasts"] == len(t)


# ------------------------------------------------------------------ metrics and bootstrap
def test_macro_mae_averages_participants_not_breakfasts():
    oof = pd.DataFrame({"participant_id": ["a"] * 3 + ["b"], "observed": [10.0, 10, 10, 10],
                        "pred_naive": [0.0, 0, 0, 0], "pred_linear": [9.0, 9, 9, 6]})
    m = ex.point_metrics(oof, "linear")
    assert m["participant_macro_mae"] == pytest.approx((1 + 4) / 2)
    assert m["pooled_mae"] == pytest.approx((1 + 1 + 1 + 4) / 4)


def test_bootstrap_resamples_whole_participants():
    W = ex.bootstrap_weights(5, 50, [1, 0])
    assert W.shape == (50, 5) and (W.sum(axis=1) == 5).all() and (W == np.round(W)).all()
    assert np.array_equal(W, ex.bootstrap_weights(5, 50, [1, 0]))
    oof = pd.DataFrame({"participant_id": ["a"] * 3 + ["b"], "observed": [10.0, 10, 10, 10],
                        "pred_naive": [0.0, 0, 0, 0], "pred_linear": [9.0, 9, 9, 6]})
    agg = ex._participant_errors(oof, ["naive", "linear"])
    s = ex.bootstrap_statistics(agg, ["naive", "linear"], np.array([[1.0, 1.0], [2.0, 0.0], [0.0, 2.0]]))
    assert np.allclose(s["linear"]["participant_macro_mae"], [2.5, 1.0, 4.0])
    assert np.allclose(s["linear"]["pooled_mae"], [7 / 4, 1.0, 4.0])             # weights all of b's breakfasts


def test_forest_rule_and_paired_comparisons():
    r = ex.run_experiment(_cfg(), _table(), use_forest=ex.random_forest_available())
    res = r.metrics["analyses"]["primary"]["dev_a"]
    lin = res["comparisons"]["linear_vs_naive"]["participant_macro_mae"]
    assert lin["point"] == pytest.approx(res["models"]["naive"]["participant_macro_mae"]
                                         - res["models"]["linear"]["participant_macro_mae"])
    assert lin["improvement"] == (lin["interval"][0] > 0)
    if ex.random_forest_available():
        rf = res["comparisons"]["random_forest_vs_linear"]["participant_macro_mae"]
        assert res["random_forest_improves_on_linear"] == (rf["interval"][0] > 0)
    ca = r.metrics["cross_analysis"]["no_baseline_glucose"]["dev_a"]
    assert ca["paired"] and "improvement" not in ca["linear"]


@pytest.mark.skipif(not ex.random_forest_available(), reason="scikit-learn not installed")
def test_deterministic_with_forest():
    a = ex.run_experiment(_cfg(), _table(), use_forest=True)
    b = ex.run_experiment(_cfg(), _table(), use_forest=True)
    assert a.metrics == b.metrics and a.model_metadata == b.model_metadata
    pd.testing.assert_frame_equal(a.oof_predictions, b.oof_predictions)


def test_runs_without_scikit_learn(monkeypatch):
    monkeypatch.setattr(ex, "random_forest_available", lambda: False)
    r = ex.run_experiment(_cfg(), _table())
    assert r.metrics["random_forest"] == {"run": False, "reason": ex.RF_UNAVAILABLE}
    assert "pred_random_forest" not in r.oof_predictions
    assert set(r.metrics["analyses"]["primary"]["dev_a"]["models"]) == {"naive", "linear"}
    assert "Random forest not run" in ex.render_report(_cfg(), r, {k: "x" for k in (
        "config_sha256", "git_commit", "analysis_table_sha256", "m1_manifest_sha256")})


# ------------------------------------------------------------------ end to end
def _task_dir(tmp_path, t, task="postprandial_breakfast"):
    d = tmp_path / "task"
    d.mkdir()
    t.to_parquet(d / "analysis_table.parquet", index=False)
    (d / "task_manifest.json").write_text(json.dumps({
        "source": {"task": task},
        "inputs": [{"path": "m1/_provenance/manifest.json", "sha256": "m1hash"}],
        "outputs": [{"path": str(d / "analysis_table.parquet"), "sha256": sha256_file(d / "analysis_table.parquet")}]}))
    (d / "task_summary.json").write_text(json.dumps({"devices": {
        dv: {"outcome_status": {"calculable": int((t.stream_id == dv).sum())}} for dv in DEVICES}}))
    return d


def test_end_to_end_outputs_lineage_and_aggregate_report(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    import yaml
    cfg_path.write_text(yaml.safe_dump(_cfg()))
    t = _table()
    res = ex.run(cfg_path, _task_dir(tmp_path, t), tmp_path / "out", tmp_path, use_forest=False)
    out = tmp_path / "out"
    for f in ("split_assignments.parquet", "oof_predictions.parquet", "metrics.json", "model_metadata.json",
              "experiment_report.md", "experiment_manifest.json"):
        assert (out / f).exists()
    man = json.loads((out / "experiment_manifest.json").read_text())
    assert man["lineage"]["m1_manifest_sha256"] == "m1hash"
    assert man["lineage"]["column_selection"]["no_baseline_glucose"] == [
        p for p in ex.APPROVED_PREDICTORS if p != "baseline_glucose_mgdl"]
    split = pd.read_parquet(out / "split_assignments.parquet")
    assert len(split) == 3 * len(DEVICES) * t.participant_id.nunique()
    report = (out / "experiment_report.md").read_text() + (out / "metrics.json").read_text()
    assert not any(p in report for p in t.participant_id.unique()) and "P0" not in report   # aggregate only
    rep = (out / "experiment_report.md").read_text()
    assert "not external validation" in rep and "does not remove" in rep and "never pooled" in rep
    assert len(res.oof_predictions) == 3 * len(t)   # 3 analyses x all rows


def test_lineage_and_count_mismatches_refuse(tmp_path):
    cfg_path = tmp_path / "cfg.yaml"
    import yaml
    cfg_path.write_text(yaml.safe_dump(_cfg()))
    d = _task_dir(tmp_path, _table())
    pd.read_parquet(d / "analysis_table.parquet").iloc[:-1].to_parquet(d / "analysis_table.parquet", index=False)
    with pytest.raises(ValueError, match="hash"):
        ex.run(cfg_path, d, tmp_path / "out", tmp_path, use_forest=False)
    t = _table()
    (tmp_path / "x").mkdir()
    d2 = _task_dir(tmp_path / "x", t)
    s = json.loads((d2 / "task_summary.json").read_text())
    s["devices"]["dev_a"]["outcome_status"]["calculable"] += 1
    (d2 / "task_summary.json").write_text(json.dumps(s))
    with pytest.raises(ValueError, match="calculable rows"):
        ex.run(cfg_path, d2, tmp_path / "out2", tmp_path, use_forest=False)
