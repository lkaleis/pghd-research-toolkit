"""M4 refinement (collinearity, fasting glucose, ridge): fold isolation, selection, join rules, outputs. Synthetic only."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from pghd_ready.conditions.cardiometabolic.experiments import breakfast_iauc as m4
from pghd_ready.conditions.cardiometabolic.experiments import breakfast_iauc_refinement as rf
from pghd_ready.core.provenance import sha256_file

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments" / "breakfast_iauc_refinement.yaml"
DEVICES = ["dev_a", "dev_b"]
REDUCED = [p for p in m4.APPROVED_PREDICTORS if p != "energy_kcal"]


def _cfg():
    cfg = rf.load_config(CONFIG)
    cfg["devices"], cfg["dataset"] = list(DEVICES), "syn"
    cfg["bootstrap"]["resamples"] = 200
    cfg["ridge"]["alphas"] = [0.01, 0.1, 1.0]
    return cfg


def _table(n_participants=7, per=5, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for p in range(n_participants):
        effect = rng.normal(0, 400)
        for b in range(per):
            c, pr, f = rng.uniform(10, 100), rng.uniform(5, 40), rng.uniform(5, 40)
            x = {"energy_kcal": 4 * c + 4 * pr + 9 * f + rng.normal(0, 20), "carbohydrate_g": c, "protein_g": pr,
                 "fat_g": f, "fiber_g": rng.uniform(0, 10), "baseline_glucose_mgdl": rng.uniform(80, 130),
                 "hba1c": 5 + p * 0.2, "body_mass_index": 22 + p * 1.3, "age": 30 + 3 * p}
            for d_i, d in enumerate(DEVICES):
                rows.append({"participant_id": f"P{p:02d}", "event_id": f"P{p:02d}-B{b}", "stream_id": d,
                             "outcome_status": "calculable",
                             "iauc_signed_mgdl_min": 40 * c - 1500 + effect + rng.normal(0, 300) + 50 * d_i, **x})
    return pd.DataFrame(rows)


def _attrs(t, unit="mg/dL", drop=None, dup=False):
    pids = sorted(t.participant_id.unique())
    a = pd.DataFrame({"participant_id": pids, "source_dataset": "syn", "concept": "fasting_glucose_lab", "unit": unit,
                      "value_num": [90.0 + 5 * i + (i % 3) * 7 for i in range(len(pids))]})
    if drop:
        a = a[a.participant_id != drop]
    if dup:
        a = pd.concat([a, a.iloc[[0]]], ignore_index=True)
    return a


def _rows(t=None, cfg=None):
    t = _table() if t is None else t
    cfg = cfg or _cfg()
    return {d: rf.add_attributes(df, _attrs(t), cfg) for d, df in m4.select_rows(t, DEVICES, "all_calculable").items()}


# ------------------------------------------------------------------ configuration
def test_shipped_config_is_the_approved_design():
    cfg = rf.load_config(CONFIG)
    mods = cfg["models"]
    assert mods["A_ols_full"] == {"type": "ols", "predictors": list(m4.APPROVED_PREDICTORS)}
    assert mods["B_ols_reduced"]["predictors"] == REDUCED
    assert mods["C_ols_reduced_fasting"]["predictors"] == REDUCED + ["fasting_glucose_lab"]
    assert (mods["D1_ridge_reduced"], mods["D2_ridge_reduced_fasting"]) == (
        {"type": "ridge", "predictors": REDUCED}, {"type": "ridge", "predictors": REDUCED + ["fasting_glucose_lab"]})
    assert np.allclose(cfg["ridge"]["alphas"], 10 ** np.arange(-3, 2.01, 0.5))
    assert cfg["ridge"]["ties"] == "larger_alpha" and cfg["ridge"]["criterion"] == "participant_macro_mae"
    assert len(cfg["comparisons"]) == 7 and cfg["reference"]["analysis"] == "primary"
    rf.validate_config(cfg, m4.load_config(ROOT / cfg["reference"]["config"]))


@pytest.mark.parametrize("mutate", [
    lambda c: c["models"]["C_ols_reduced_fasting"]["predictors"].append("fasting_insulin"),   # no other biomarker
    lambda c: c["models"]["B_ols_reduced"]["predictors"].append("n_outcome_readings"),
    lambda c: c["models"]["A_ols_full"]["predictors"].remove("energy_kcal"),                 # reference fixed
    lambda c: c.update(added_attributes={"fasting_glucose_lab": "mmol/L"}),
    lambda c: c["ridge"].update(ties="smaller_alpha"),
    lambda c: c["ridge"].update(alphas=[1.0, 0.1]),
    lambda c: c["validation"].update(scheme="kfold"),
    lambda c: c.update(target="iauc_positive"),
    lambda c: c["comparisons"].append({"candidate": "B_ols_reduced", "reference": "B_ols_reduced"}),
])
def test_config_guards(mutate):
    cfg = _cfg()
    mutate(cfg)
    with pytest.raises(ValueError):
        rf.validate_config(cfg)


def test_reference_settings_must_match():
    cfg, ref = _cfg(), m4.load_config(ROOT / "configs" / "experiments" / "breakfast_iauc.yaml")
    ref["devices"] = list(DEVICES)
    ref["bootstrap"] = dict(cfg["bootstrap"])
    rf.validate_config(cfg, ref)
    with pytest.raises(ValueError, match="bootstrap"):
        rf.validate_config(cfg, dict(ref, bootstrap=dict(cfg["bootstrap"], seed=1)))
    with pytest.raises(ValueError, match="order"):
        rf.validate_config(cfg, dict(ref, devices=DEVICES[::-1]))


# ------------------------------------------------------------------ fasting glucose join
def test_fasting_glucose_join_rules():
    t, cfg = _table(), _cfg()
    df = m4.select_rows(t, DEVICES, "all_calculable")["dev_a"]
    out = rf.add_attributes(df, _attrs(t), cfg)
    assert len(out) == len(df) and out.groupby("participant_id")["fasting_glucose_lab"].nunique().eq(1).all()
    with pytest.raises(ValueError, match="another unit"):
        rf.add_attributes(df, _attrs(t, unit="mmol/L"), cfg)
    with pytest.raises(ValueError, match="no imputation"):
        rf.add_attributes(df, _attrs(t, drop="P03"), cfg)
    with pytest.raises(ValueError, match="more than once"):
        rf.add_attributes(df, _attrs(t, dup=True), cfg)


# ------------------------------------------------------------------ ridge
def test_ridge_limits_and_unpenalised_intercept():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(40, 3)) * [1, 10, 100] + [5, 50, 500]
    y = X @ [2.0, -0.3, 0.01] + rng.normal(size=40)
    ols = m4._ols_predict(m4._ols_fit(X, y), X)
    assert np.allclose(rf.ridge_predict(rf.ridge_fit(X, y, 1e-10), X), ols, atol=1e-6)
    assert np.allclose(rf.ridge_predict(rf.ridge_fit(X, y, 1e10), X), y.mean(), atol=1e-6)


def test_ties_resolve_to_the_larger_alpha():
    X = np.random.default_rng(2).normal(size=(12, 2))
    g = np.repeat(["a", "b", "c", "d"], 3)
    alpha, scores = rf.select_alpha(X, np.full(12, 5.0), g, [0.01, 0.1, 1.0])
    assert alpha == 1.0 and len(set(scores)) == 1


def test_inner_selection_and_scaling_never_see_the_inner_held_out_participant(monkeypatch):
    X = np.random.default_rng(3).normal(size=(15, 2))
    y = X @ [1.0, 2.0] + np.random.default_rng(4).normal(size=15)
    g = np.repeat(list("abcde"), 3)
    calls = []
    fit, predict = rf.ridge_fit, rf.ridge_predict
    monkeypatch.setattr(rf, "ridge_fit", lambda Xf, yf, a: calls.append(("fit", {tuple(r) for r in Xf})) or fit(Xf, yf, a))
    monkeypatch.setattr(rf, "ridge_predict", lambda f, Xp: calls.append(("pred", {tuple(r) for r in Xp})) or predict(f, Xp))
    rf.select_alpha(X, y, g, [0.1, 1.0])
    pairs = list(zip(calls[::2], calls[1::2]))
    assert len(pairs) == 5 * 2 and all(a[0] == "fit" and b[0] == "pred" and not (a[1] & b[1]) for a, b in pairs)


def test_outer_held_out_participant_never_influences_its_own_fold():
    cfg, t = _cfg(), _table()
    base_oof, base_coef = rf.lopo(_rows(t, cfg)["dev_a"], cfg)
    t2 = t.copy()
    sel = t2.participant_id == "P02"
    t2.loc[sel, "iauc_signed_mgdl_min"] += 1e5
    t2.loc[sel, ["carbohydrate_g", "protein_g", "baseline_glucose_mgdl"]] *= 7
    alt_oof, alt_coef = rf.lopo(_rows(t2, cfg)["dev_a"], cfg)
    k = int(base_oof.loc[base_oof.participant_id == "P02", "fold"].iloc[0])
    pd.testing.assert_frame_equal(base_coef[base_coef.fold == k].reset_index(drop=True),
                                  alt_coef[alt_coef.fold == k].reset_index(drop=True))   # alpha, scaling, fit unchanged
    assert not base_coef[base_coef.fold != k].equals(alt_coef[alt_coef.fold != k])       # it is training data elsewhere


# ------------------------------------------------------------------ experiment
def test_model_a_and_naive_reproduce_m4():
    cfg, t = _cfg(), _table()
    res = rf.run_experiment(cfg, _rows(t, cfg))
    m4_cfg = m4.load_config(ROOT / "configs" / "experiments" / "breakfast_iauc.yaml")
    m4_cfg["devices"] = list(DEVICES)
    ref = m4.run_experiment(m4_cfg, t, use_forest=False).oof_predictions
    assert rf.check_reference(res, ref, DEVICES) == {"dev_a": 0.0, "dev_b": 0.0}
    broken = ref.copy()
    broken.loc[broken.index[0], "pred_linear"] += 1.0
    with pytest.raises(ValueError, match="differ from M4"):
        rf.check_reference(res, broken, DEVICES)


def test_paired_improvement_sign_and_shared_resamples():
    res = rf.run_experiment(_cfg(), _rows())
    d = res.metrics["devices"]["dev_a"]
    for c in d["comparisons"]:
        e = c["participant_macro_mae"]
        assert e["point"] == pytest.approx(d["models"][c["reference"]]["participant_macro_mae"]
                                           - d["models"][c["candidate"]]["participant_macro_mae"])
        assert e["verdict"] in {"candidate lower error", "candidate higher error", "no detectable difference"}
    assert res.metrics["improvement_definition"].startswith("MAE(reference) - MAE(candidate)")


def test_devices_never_pooled_and_deterministic():
    cfg, t = _cfg(), _table()
    a = rf.run_experiment(cfg, _rows(t, cfg))
    assert a.metrics == rf.run_experiment(cfg, _rows(t, cfg)).metrics
    t2 = t.copy()
    t2.loc[t2.stream_id == "dev_b", "iauc_signed_mgdl_min"] *= 2
    b = rf.run_experiment(cfg, _rows(t2, cfg))
    assert a.metrics["devices"]["dev_a"] == b.metrics["devices"]["dev_a"]
    assert a.metrics["devices"]["dev_b"] != b.metrics["devices"]["dev_b"]
    assert set(a.metrics["devices"]) == set(DEVICES)


def test_diagnostics_vif_alpha_and_stability():
    res = rf.run_experiment(_cfg(), _rows())
    diag = res.model_metadata["devices"]["dev_a"]
    assert diag["vif"]["A_ols_full"]["energy_kcal"] > diag["vif"]["B_ols_reduced"]["carbohydrate_g"]
    assert set(diag["vif"]) == {"A_ols_full", "B_ols_reduced", "C_ols_reduced_fasting"}
    assert sum(diag["selected_alpha_counts"]["D1_ridge_reduced"].values()) == 7
    s = diag["coefficients"]["C_ols_reduced_fasting"]["fasting_glucose_lab"]
    assert s["fold_min"] <= s["fold_median"] <= s["fold_max"] and 0 <= s["share_same_sign_as_full"] <= 1


# ------------------------------------------------------------------ end to end
def _write_inputs(tmp_path, t):
    m1 = tmp_path / "m1"
    (m1 / "_provenance").mkdir(parents=True)
    _attrs(t).to_parquet(m1 / "participant_attributes.parquet", index=False)
    (m1 / "_provenance" / "manifest.json").write_text(json.dumps({"outputs": [
        {"path": "m1/participant_attributes.parquet", "sha256": sha256_file(m1 / "participant_attributes.parquet")}]}))
    task = tmp_path / "task"
    task.mkdir()
    t.to_parquet(task / "analysis_table.parquet", index=False)
    (task / "task_manifest.json").write_text(json.dumps({
        "source": {"task": "postprandial_breakfast"},
        "inputs": [{"path": "m1/_provenance/manifest.json", "sha256": sha256_file(m1 / "_provenance" / "manifest.json")}],
        "outputs": [{"path": "task/analysis_table.parquet", "sha256": sha256_file(task / "analysis_table.parquet")}]}))
    (task / "task_summary.json").write_text(json.dumps({"devices": {
        d: {"outcome_status": {"calculable": int((t.stream_id == d).sum())}} for d in DEVICES}}))
    cfg = _cfg()
    m4_cfg = m4.load_config(ROOT / "configs" / "experiments" / "breakfast_iauc.yaml")
    m4_cfg["devices"], m4_cfg["bootstrap"] = list(DEVICES), dict(cfg["bootstrap"])
    (tmp_path / "m4.yaml").write_text(yaml.safe_dump(m4_cfg))
    m4.run(tmp_path / "m4.yaml", task, tmp_path / "m4_out", tmp_path, use_forest=False)
    cfg["reference"]["config"] = "m4.yaml"
    (tmp_path / "ref.yaml").write_text(yaml.safe_dump(cfg))
    return tmp_path / "ref.yaml", task, m1, tmp_path / "m4_out"


def test_end_to_end_outputs_lineage_and_reference_untouched(tmp_path):
    t = _table()
    cfg_path, task, m1, ref = _write_inputs(tmp_path, t)
    before = {p.name: sha256_file(p) for p in ref.iterdir()}
    out = tmp_path / "out"
    rf.run(cfg_path, task, m1, ref, out, tmp_path)
    assert before == {p.name: sha256_file(p) for p in ref.iterdir()}                       # M4 outputs unchanged
    for f in ("split_assignments.parquet", "oof_predictions.parquet", "fold_coefficients.parquet", "metrics.json",
              "model_metadata.json", "refinement_report.md", "refinement_manifest.json"):
        assert (out / f).exists()
    man = json.loads((out / "refinement_manifest.json").read_text())
    assert man["lineage"]["reference_check_max_abs_diff"] == {"dev_a": 0.0, "dev_b": 0.0}
    rep = (out / "refinement_report.md").read_text()
    agg = rep + (out / "metrics.json").read_text() + (out / "model_metadata.json").read_text()
    assert not any(p in agg for p in t.participant_id.unique())
    for phrase in ("not external validation", "not** the per-breakfast baseline glucose", "cannot be verified",
                   "not independently verifiable", "never pooled", "not evidence of equivalence"):
        assert phrase in rep
    with pytest.raises(ValueError, match="must not write"):
        rf.run(cfg_path, task, m1, ref, ref, tmp_path)


def test_lineage_mismatch_refuses(tmp_path):
    cfg_path, task, m1, ref = _write_inputs(tmp_path, _table())
    man = json.loads((ref / "experiment_manifest.json").read_text())
    man["lineage"]["analysis_table_sha256"] = "0" * 64
    (ref / "experiment_manifest.json").write_text(json.dumps(man))
    with pytest.raises(ValueError, match="different analysis table"):
        rf.run(cfg_path, task, m1, ref, tmp_path / "out", tmp_path)
    _attrs(_table(), unit="mg/dL").assign(value_num=1.0).to_parquet(m1 / "participant_attributes.parquet", index=False)
    with pytest.raises(ValueError, match="lineage broken"):
        rf.run(cfg_path, task, m1, ref, tmp_path / "out", tmp_path)
