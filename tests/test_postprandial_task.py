"""M3 postprandial breakfast task: outcome definition, calculability, separation, end-to-end. Synthetic only."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from pghd_ready.cli import main as cli_main
from pghd_ready.conditions.cardiometabolic.tasks import postprandial as pp

from conftest import REPO_ROOT

TASK = REPO_ROOT / "configs" / "tasks" / "postprandial_breakfast.yaml"
T0 = pd.Timestamp("2024-01-01 08:00:00")
DEVICES = ["libre_pro", "dexcom_g6"]


def _spec(min_readings=3, max_gap_min=5):
    s = copy.deepcopy(pp.load_task(TASK))
    s["dataset"] = "syn"
    s["baseline"]["min_readings"] = {d: min_readings for d in DEVICES}
    s["outcome"]["max_gap_min"] = {d: max_gap_min for d in DEVICES}
    return s


def _glucose(pid, stream, minutes, values, day=0, unit="mg/dL"):
    base = T0 + pd.Timedelta(days=day)
    return pd.DataFrame({"participant_id": pid, "source_dataset": "syn", "stream_id": stream,
                         "concept": "glucose_interstitial", "timestamp": [base + pd.Timedelta(minutes=m) for m in minutes],
                         "value": [float(v) for v in values], "unit": unit, "raw_value": "x",
                         "is_interpolated": pd.array([pd.NA] * len(minutes), dtype="boolean"),
                         "quality_flags": [[] for _ in minutes], "source_file": f"{pid}.csv",
                         "source_row": pd.array(range(len(minutes)), dtype="Int64"), "source_column": stream})


def _meal(eid, pid, minute, subtype="breakfast", day=0):
    return {"event_id": eid, "participant_id": pid, "source_dataset": "syn", "event_type": "meal",
            "event_subtype": subtype, "start_time": T0 + pd.Timedelta(days=day, minutes=minute),
            "energy_kcal": 300.0, "carbohydrate_g": 40.0, "protein_g": 20.0, "fat_g": 8.0, "fiber_g": 2.0,
            "amount_consumed_raw": 100.0}


def _attrs(pids):
    rows = []
    for p in pids:
        rows += [{"participant_id": p, "source_dataset": "syn", "concept": c, "unit": u, "value_num": v}
                 for c, u, v in (("hba1c", "%", 5.8), ("body_mass_index", "kg/m2", 27.0), ("age", "a", 50.0))]
    return pd.DataFrame(rows)


def _flat_then_rise(pid="p1", baseline=100.0, day=0):
    """Baseline 100 for [-15, 0); then rises linearly 100 -> 160 over 0..60 and stays at 160 to 120 (1-min)."""
    mins = list(range(-15, 121))
    vals = [baseline if m < 0 else (100 + m if m <= 60 else 160) for m in mins]
    return pd.concat([_glucose(pid, d, mins, vals, day=day) for d in DEVICES], ignore_index=True)


def _run(obs, meals, spec=None):
    return pp.run_task(spec or _spec(), obs, pd.DataFrame(meals), _attrs(sorted({m["participant_id"] for m in meals})))


def _row(res, device="dexcom_g6", eid="e1"):
    t = res.analysis_table
    return t[(t.stream_id == device) & (t.event_id == eid)].iloc[0]


# ------------------------------------------------------------------ outcome definition
def test_signed_iauc_matches_the_analytic_value():
    r = _row(_run(_flat_then_rise(), [_meal("e1", "p1", 0)]))
    # integral of (g - 100): triangle 0..60 (0 -> 60) = 1800, plus 60 x 60 = 3600 -> 5400 mg/dL x min
    assert r.outcome_status == "calculable" and r.iauc_signed_mgdl_min == pytest.approx(5400)
    assert r.baseline_glucose_mgdl == pytest.approx(100) and r.n_baseline_readings == 15 and r.n_outcome_readings == 121


def test_below_baseline_is_negative_not_clamped():
    mins = list(range(-15, 121))
    obs = pd.concat([_glucose("p1", d, mins, [100 if m < 0 else 90 for m in mins]) for d in DEVICES])
    r = _row(_run(obs, [_meal("e1", "p1", 0)]))
    assert r.iauc_signed_mgdl_min == pytest.approx(-10 * 120)


def test_baseline_window_excludes_t0_and_uses_mean():
    mins = list(range(-15, 121))
    vals = [80 if m < -10 else (120 if m < 0 else 100) for m in mins]   # reading at t0 = 100 is not in the baseline
    obs = pd.concat([_glucose("p1", d, mins, vals) for d in DEVICES])
    r = _row(_run(obs, [_meal("e1", "p1", 0)]))
    assert r.baseline_glucose_mgdl == pytest.approx((5 * 80 + 10 * 120) / 15)


# ------------------------------------------------------------------ calculability, no fallbacks
def _drop(obs, minutes, device="dexcom_g6"):
    off = (obs["timestamp"] - T0).dt.total_seconds() / 60
    return obs[~((obs["stream_id"] == device) & off.isin(minutes))]


@pytest.mark.parametrize("dropped, reason", [
    (list(range(-15, -1)), "insufficient_baseline_readings"),      # 1 baseline reading left, 3 required
    (list(range(-15, 0)), "insufficient_baseline_readings"),       # none left: no substitute baseline
    ([0], "missing_reading_at_window_start"),
    ([120], "missing_reading_at_window_end"),
    (list(range(30, 37)), "gap_exceeds_max"),                       # 8-min interval > 5 min
    (list(range(0, 121)), "no_outcome_readings"),
])
def test_not_calculable_reasons(dropped, reason):
    res = _run(_drop(_flat_then_rise(), dropped), [_meal("e1", "p1", 0)])
    r = _row(res)
    assert (r.outcome_status, r.outcome_reason) == ("not_calculable", reason) and np.isnan(r.iauc_signed_mgdl_min)
    assert np.isnan(r.baseline_glucose_mgdl)
    assert _row(res, device="libre_pro").outcome_status == "calculable"     # devices are independent


def test_gap_equal_to_max_is_allowed():
    r = _row(_run(_drop(_flat_then_rise(), list(range(31, 35))), [_meal("e1", "p1", 0)]))  # 5-min interval
    assert r.outcome_status == "calculable"


def test_unit_mismatch_and_duplicate_timestamp_are_not_calculable():
    obs = _flat_then_rise()
    other = obs.copy()
    other.loc[(other.stream_id == "dexcom_g6"), "unit"] = "mmol/L"
    assert _row(_run(other, [_meal("e1", "p1", 0)])).outcome_reason == "unit_mismatch"
    dup = pd.concat([obs, obs[(obs.stream_id == "dexcom_g6")].iloc[[20]]], ignore_index=True)
    assert _row(_run(dup, [_meal("e1", "p1", 0)])).outcome_reason == "duplicate_timestamp"


def test_undecided_parameters_mean_not_computed():
    spec = _spec(min_readings=None, max_gap_min=None)
    r = _row(_run(_flat_then_rise(), [_meal("e1", "p1", 0)], spec=spec))
    assert (r.outcome_status, r.outcome_reason) == ("not_computed", "parameter_undecided")
    assert np.isnan(r.iauc_signed_mgdl_min) and np.isnan(r.baseline_glucose_mgdl)


def test_shipped_task_parameters_are_the_approved_values():
    spec = pp.load_task(TASK)
    assert spec["baseline"]["min_readings"] == {"libre_pro": 1, "dexcom_g6": 3}
    assert spec["outcome"]["max_gap_min"] == {"libre_pro": 15, "dexcom_g6": 5}
    assert (spec["baseline"]["start_min"], spec["baseline"]["end_min"]) == (-15, 0)
    assert (spec["outcome"]["start_min"], spec["outcome"]["end_min"]) == (0, 120)
    assert spec["sampling_origin"]["policy"] == "native_and_unknown"
    assert "interpolated" in spec["sampling_origin"]["note"] and "never as native" in spec["sampling_origin"]["note"]


def test_task_config_must_list_every_device(tmp_path):
    text = TASK.read_text()
    assert "    dexcom_g6: 3\n" in text
    text = text.replace("    dexcom_g6: 3\n", "", 1)
    p = tmp_path / "t.yaml"
    p.write_text(text)
    with pytest.raises(ValueError, match="must list every device"):
        pp.load_task(p)


# ------------------------------------------------------------------ separation and near duplicates
def test_meal_separation_boundaries_and_near_duplicates_are_flagged_not_merged():
    obs = pd.concat([_flat_then_rise("p1", day=d) for d in range(4)], ignore_index=True)
    meals = [_meal("e_ok", "p1", 0, day=0),
             _meal("e_before", "p1", 0, day=1), _meal("s1", "p1", -60, "snack", day=1),       # at -60: excluded
             _meal("e_edge", "p1", 0, day=2), _meal("s2", "p1", -61, "snack", day=2),          # at -61: kept
             _meal("e_dup", "p1", 0, day=3), _meal("e_dup2", "p1", 2, "breakfast", day=3)]     # 2 min apart
    res = _run(obs, meals)
    led = res.ledger[res.ledger.stream_id == "dexcom_g6"].set_index("event_id")
    assert led.loc["e_ok", "status"] == "eligible" and led.loc["e_edge", "status"] == "eligible"
    assert led.loc["e_before", "excluded_by"] == "no_meal_before"
    assert led.loc["e_dup", "excluded_by"] == "no_meal_after" and led.loc["e_dup2", "excluded_by"] == "no_meal_before"
    assert led.loc["e_dup", "near_duplicate_flag"] and led.loc["e_dup2", "near_duplicate_flag"]
    assert {"e_dup", "e_dup2"} <= set(led.index)                       # both kept as separate candidates
    assert len(res.ledger) == 2 * 5                                    # 5 breakfasts x 2 devices
    s = res.summary["devices"]["dexcom_g6"]
    assert s["candidates"] == 5 and s["eligible"] == 2 and s["candidates_near_duplicate_flagged"] == 2


def test_device_without_readings_is_excluded_for_that_device_only():
    obs = _flat_then_rise()
    obs = obs[obs.stream_id == "libre_pro"]
    res = _run(obs, [_meal("e1", "p1", 0)])
    led = res.ledger.set_index("stream_id")
    assert led.loc["dexcom_g6", "excluded_by"] == "device_readings_present" and led.loc["libre_pro", "status"] == "eligible"


def test_predictors_and_no_excluded_inputs():
    t = _run(_flat_then_rise(), [_meal("e1", "p1", 0)]).analysis_table
    for c in ("energy_kcal", "carbohydrate_g", "protein_g", "fat_g", "fiber_g", "baseline_glucose_mgdl",
              "hba1c", "body_mass_index", "age"):
        assert c in t.columns
    assert "amount_consumed_raw" not in t.columns and "start_time" not in t.columns
    assert (t["n_origin_unknown"] > 0).all() and (t["n_origin_native"] == 0).all()


# ------------------------------------------------------------------ end to end
def test_prepare_cli_end_to_end(tmp_path):
    m1 = tmp_path / "m1"
    (m1 / "_provenance").mkdir(parents=True)
    obs = _flat_then_rise()
    meals = pd.DataFrame([_meal("e1", "p1", 0)])
    obs.to_parquet(m1 / "observations.parquet", index=False)
    meals.to_parquet(m1 / "events.parquet", index=False)
    _attrs(["p1"]).to_parquet(m1 / "participant_attributes.parquet", index=False)
    (m1 / "_provenance" / "manifest.json").write_text("{}")
    task = tmp_path / "task.yaml"
    task.write_text(TASK.read_text().replace("dataset: cgmacros", "dataset: syn"))
    before = {p.name: p.read_bytes() for p in m1.glob("*.parquet")}
    for out in ("o1", "o2"):
        assert cli_main(["prepare", "--task", str(task), "--m1", str(m1), "--out", str(tmp_path / out),
                         "--base-dir", str(tmp_path)]) == 0
    assert {p.name: p.read_bytes() for p in m1.glob("*.parquet")} == before
    for name in ("ledger", "funnel", "windows", "analysis_table"):
        assert (tmp_path / "o1" / f"{name}.parquet").read_bytes() == (tmp_path / "o2" / f"{name}.parquet").read_bytes()
    report = (tmp_path / "o1" / "task_report.md").read_text()
    assert "p1" not in report and "e1" not in report and "Signed incremental AUC" in report
    assert "not clinically validated" in report
    manifest = json.loads((tmp_path / "o1" / "task_manifest.json").read_text())
    assert any(i["path"] == "task.yaml" for i in manifest["inputs"]) and "task" in manifest["configs"]
