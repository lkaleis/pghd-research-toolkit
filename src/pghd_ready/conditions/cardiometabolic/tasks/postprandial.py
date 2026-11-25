"""Postprandial glucose-response task (M3 reference workflow): analytical preparation, no modelling.

Research-specific definitions (owner decisions, 2026-11-25) implemented here, on top of the generic
:mod:`pghd_ready.core.alignment` and :mod:`pghd_ready.core.eligibility`:

* **Candidates:** index meals (configured type/subtype) x CGM device; devices are never pooled.
* **Eligibility** (in order): no other meal in [t0 - before, t0); no other meal in [t0, t0 + after];
  the participant has readings from the device. Meals within ``near_duplicate_flag_min`` of another meal are
  *flagged* as potentially duplicated or ambiguous; they are never merged.
* **Baseline:** mean glucose of the readings in [t0 + start, t0 + end) (default [t0 - 15 min, t0)).
  Requires at least ``baseline.min_readings[device]`` readings; there is no fallback definition.
* **Outcome:** signed incremental AUC over [t0, t0 + 120 min]: the trapezoidal integral of
  (glucose - baseline) over the readings in the window, in mg/dL x min. Values below baseline contribute
  negatively; nothing is clamped. The integral needs readings exactly at both window ends (no
  extrapolation), and no interval between consecutive readings longer than ``outcome.max_gap_min[device]``
  (the trapezoid implies a straight line across each interval).
* If a required parameter is undecided (null), outcomes are not computed (``parameter_undecided``).
* Sampling origin is carried through unchanged; readings are admitted only by the configured policy.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import pandera.pandas as pa
import yaml

from ....core import alignment as al
from ....core import eligibility as el
from ....core.provenance import RunManifest, config_hash
from ....core.sampling import InclusionPolicy

CRITERIA = ["no_meal_before", "no_meal_after", "device_readings_present"]
OUTCOME_REASONS = ["parameter_undecided", "unit_mismatch", "value_missing", "duplicate_timestamp",
                   "insufficient_baseline_readings", "no_outcome_readings", "missing_reading_at_window_start",
                   "missing_reading_at_window_end", "gap_exceeds_max"]


@dataclass
class TaskResult:
    ledger: pd.DataFrame          # one row per candidate (meal x device), participant-level
    funnel: pd.DataFrame          # aggregate, per device
    windows: pd.DataFrame         # readings in baseline/outcome windows, participant-level
    analysis_table: pd.DataFrame  # one row per eligible meal x device, participant-level
    summary: dict[str, Any]       # aggregate only


def load_task(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        spec = yaml.safe_load(fh)
    for key in ("task", "version", "dataset", "index_events", "separation", "devices", "glucose", "sampling_origin",
                "baseline", "outcome", "predictors"):
        if key not in spec:
            raise ValueError(f"Task config: missing {key!r}.")
    InclusionPolicy(spec["sampling_origin"]["policy"])  # must be explicit and valid
    for block, key in (("baseline", "min_readings"), ("outcome", "max_gap_min")):
        values = spec[block].get(key)
        if not isinstance(values, dict) or set(values) != set(spec["devices"]):
            raise ValueError(f"Task config: {block}.{key} must list every device (null = undecided).")
    return spec


def _select(events: pd.DataFrame, sel: Mapping[str, str]) -> pd.DataFrame:
    m = pd.Series(True, index=events.index)
    for col, val in sel.items():
        m &= events[col] == val
    return events[m]


def signed_iauc(offsets_s: np.ndarray, values: np.ndarray, baseline: float) -> float:
    """Trapezoidal integral of (value - baseline) over the given readings, in value-unit x minutes."""
    t = np.asarray(offsets_s, dtype=float) / 60.0
    y = np.asarray(values, dtype=float) - baseline
    return float(np.sum((y[1:] + y[:-1]) / 2.0 * np.diff(t)))


def _outcome(win: pd.DataFrame, device: str, spec: dict) -> dict[str, Any]:
    """Baseline and outcome for one candidate; never substitutes a definition or fills a value."""
    b_spec, o_spec, unit = spec["baseline"], spec["outcome"], spec["glucose"]["unit"]
    base = win[win["window"] == "baseline"]
    post = win[win["window"] == "outcome"]
    o_start, o_end = o_spec["start_min"] * 60.0, o_spec["end_min"] * 60.0
    gaps = np.diff(post["offset_s"].to_numpy()) if len(post) > 1 else np.array([])
    r: dict[str, Any] = {
        "n_baseline_readings": len(base), "n_outcome_readings": len(post),
        "outcome_max_interval_s": float(gaps.max()) if len(gaps) else np.nan,
        "has_reading_at_window_start": bool((post["offset_s"] == o_start).any()),
        "has_reading_at_window_end": bool((post["offset_s"] == o_end).any()),
        "n_origin_native": int((win["sampling_origin"] == "native").sum()),
        "n_origin_interpolated": int((win["sampling_origin"] == "interpolated").sum()),
        "n_origin_unknown": int((win["sampling_origin"] == "unknown").sum()),
        "baseline_glucose_mgdl": np.nan, "iauc_signed_mgdl_min": np.nan,
        "outcome_status": "not_calculable", "outcome_reason": None,
    }
    min_n, max_gap = b_spec["min_readings"][device], o_spec["max_gap_min"][device]

    def fail(reason):
        r["outcome_reason"] = reason
        return r

    if min_n is None or max_gap is None:
        r["outcome_status"] = "not_computed"
        return fail("parameter_undecided")
    if (win["unit"] != unit).any():
        return fail("unit_mismatch")
    if win["value"].isna().any():
        return fail("value_missing")
    if base["timestamp"].duplicated().any() or post["timestamp"].duplicated().any():
        return fail("duplicate_timestamp")
    if len(base) < int(min_n) or len(base) == 0:
        return fail("insufficient_baseline_readings")
    if post.empty:
        return fail("no_outcome_readings")
    if not r["has_reading_at_window_start"]:
        return fail("missing_reading_at_window_start")
    if not r["has_reading_at_window_end"]:
        return fail("missing_reading_at_window_end")
    if len(gaps) and gaps.max() > float(max_gap) * 60.0:
        return fail("gap_exceeds_max")
    baseline = float(base["value"].mean())
    r["baseline_glucose_mgdl"] = baseline
    r["iauc_signed_mgdl_min"] = signed_iauc(post["offset_s"].to_numpy(), post["value"].to_numpy(), baseline)
    r["outcome_status"], r["outcome_reason"] = "calculable", None
    return r


def analysis_schema(predictor_cols: list[str]) -> pa.DataFrameSchema:
    cols = {
        "participant_id": pa.Column(str), "event_id": pa.Column(str, unique=False), "stream_id": pa.Column(str),
        "near_duplicate_flag": pa.Column(bool),
        "outcome_status": pa.Column(str, pa.Check.isin(["calculable", "not_calculable", "not_computed"])),
        "outcome_reason": pa.Column(object, nullable=True),
        "iauc_signed_mgdl_min": pa.Column(float, nullable=True),
        "baseline_glucose_mgdl": pa.Column(float, nullable=True),
        "n_baseline_readings": pa.Column(int, pa.Check.ge(0)), "n_outcome_readings": pa.Column(int, pa.Check.ge(0)),
        "n_origin_native": pa.Column(int), "n_origin_interpolated": pa.Column(int), "n_origin_unknown": pa.Column(int),
    }
    cols.update({c: pa.Column(float, nullable=True) for c in predictor_cols})
    return pa.DataFrameSchema(cols, unique=["event_id", "stream_id"], checks=[
        pa.Check(lambda d: (d["outcome_status"] == "calculable") == d["iauc_signed_mgdl_min"].notna(),
                 element_wise=False, error="an outcome value exists exactly when the status is calculable"),
        pa.Check(lambda d: d["outcome_reason"].isna() == (d["outcome_status"] == "calculable"),
                 element_wise=False, error="every non-calculable outcome has a reason")])


def run_task(spec: dict, observations: pd.DataFrame, events: pd.DataFrame,
             attributes: pd.DataFrame) -> TaskResult:
    ds, devices = spec["dataset"], list(spec["devices"])
    events = events[events["source_dataset"] == ds]
    obs = observations[(observations["source_dataset"] == ds) & observations["stream_id"].isin(devices)
                       & (observations["concept"] == spec["glucose"]["concept"])]
    index = _select(events, spec["index_events"]).sort_values("event_id", kind="mergesort")
    sep = spec["separation"]
    nb = al.neighbouring_events(index, _select(events, sep["events"]), sep["before_min"] * 60, sep["after_min"] * 60)
    meals = index[["event_id", "participant_id", "start_time"] + spec["predictors"]["event"]].merge(nb, on="event_id")
    meals["near_duplicate_flag"] = meals["nearest_abs_s"] <= spec["near_duplicate_flag_min"] * 60

    present = obs.groupby(["participant_id", "stream_id"]).size()
    cand = meals.merge(pd.DataFrame({"stream_id": devices}), how="cross")
    cand = cand.sort_values(["stream_id", "event_id"], kind="mergesort").reset_index(drop=True)
    has = pd.Series([(p, s) in present.index for p, s in zip(cand["participant_id"], cand["stream_id"])],
                    index=cand.index)
    ledger = el.apply_criteria(cand, [("no_meal_before", cand["n_before"] == 0),
                                      ("no_meal_after", cand["n_after"] == 0),
                                      ("device_readings_present", has)])
    fun = el.funnel(ledger, CRITERIA, by=["stream_id"])

    eligible = ledger[ledger["status"] == el.ELIGIBLE]
    b, o = spec["baseline"], spec["outcome"]
    windows = al.extract_windows(
        eligible.drop_duplicates("event_id")[["event_id", "participant_id", "start_time"]], obs, devices,
        [al.Window("baseline", b["start_min"] * 60, b["end_min"] * 60, closed="left"),
         al.Window("outcome", o["start_min"] * 60, o["end_min"] * 60, closed="both")],
        spec["sampling_origin"]["policy"])
    windows = windows.merge(eligible[["event_id", "stream_id"]], on=["event_id", "stream_id"])  # eligible pairs only

    by_key = {k: g for k, g in windows.groupby(["event_id", "stream_id"])}
    empty = windows.iloc[0:0]
    outcomes = pd.DataFrame([{"event_id": e, "stream_id": s, **_outcome(by_key.get((e, s), empty), s, spec)}
                             for e, s in zip(eligible["event_id"], eligible["stream_id"])])
    attrs = al.attributes_wide(attributes, spec["predictors"]["attributes"], dataset=ds)
    table = eligible[["participant_id", "event_id", "stream_id", "near_duplicate_flag"]
                     + spec["predictors"]["event"]].merge(outcomes, on=["event_id", "stream_id"], how="left")
    table = table.merge(attrs, on="participant_id", how="left")
    predictor_cols = spec["predictors"]["event"] + ["baseline_glucose_mgdl"] + list(spec["predictors"]["attributes"])
    for c in predictor_cols:
        table[c] = table[c].astype(float)
    ordered = (["participant_id", "event_id", "stream_id", "near_duplicate_flag", "outcome_status", "outcome_reason",
                "iauc_signed_mgdl_min"] + predictor_cols
               + ["n_baseline_readings", "n_outcome_readings", "outcome_max_interval_s", "has_reading_at_window_start",
                  "has_reading_at_window_end", "n_origin_native", "n_origin_interpolated", "n_origin_unknown"])
    table = table[ordered].sort_values(["stream_id", "event_id"], kind="mergesort").reset_index(drop=True)
    table = analysis_schema([c for c in predictor_cols if c != "baseline_glucose_mgdl"]).validate(table)
    return TaskResult(ledger=ledger, funnel=fun, windows=windows, analysis_table=table,
                      summary=summarise(ledger, table, spec, attrs.attrs.get("unit_mismatch", {})))


def summarise(ledger: pd.DataFrame, table: pd.DataFrame, spec: dict, unit_mismatch: Mapping[str, int]) -> dict:
    out: dict[str, Any] = {"devices": {}}
    for dev in spec["devices"]:
        lg, t = ledger[ledger["stream_id"] == dev], table[table["stream_id"] == dev]
        reasons = t["outcome_reason"].value_counts().reindex(OUTCOME_REASONS, fill_value=0)
        out["devices"][dev] = {
            "candidates": int(len(lg)), "eligible": int((lg["status"] == el.ELIGIBLE).sum()),
            "excluded_by": {c: int((lg["excluded_by"] == c).sum()) for c in CRITERIA},
            "candidates_near_duplicate_flagged": int(lg["near_duplicate_flag"].sum()),
            "outcome_status": {s: int((t["outcome_status"] == s).sum())
                               for s in ("calculable", "not_calculable", "not_computed")},
            "outcome_reasons": {k: int(v) for k, v in reasons.items() if v},
            "window_reading_origin": {k: int(t[f"n_origin_{k}"].sum()) for k in ("native", "interpolated", "unknown")},
            "predictor_missing": {c: int(t[c].isna().sum()) for c in
                                  spec["predictors"]["event"] + list(spec["predictors"]["attributes"])},
            "descriptive_eligible_windows": {
                "baseline_readings_median": _q(t["n_baseline_readings"], 0.5),
                "baseline_readings_min": _q(t["n_baseline_readings"], 0.0),
                "outcome_max_interval_s_median": _q(t["outcome_max_interval_s"], 0.5),
                "outcome_max_interval_s_max": _q(t["outcome_max_interval_s"], 1.0),
                "with_reading_at_window_start": int(t["has_reading_at_window_start"].sum()),
                "with_reading_at_window_end": int(t["has_reading_at_window_end"].sum()),
            },
        }
        calc = t.loc[t["outcome_status"] == "calculable", "iauc_signed_mgdl_min"]
        if len(calc):
            out["devices"][dev]["iauc_signed_mgdl_min"] = {
                "n": int(len(calc)), "mean": float(calc.mean()), "sd": float(calc.std(ddof=1)) if len(calc) > 1 else None,
                "p25": float(calc.quantile(0.25)), "median": float(calc.median()), "p75": float(calc.quantile(0.75)),
                "min": float(calc.min()), "max": float(calc.max()), "n_negative": int((calc < 0).sum())}
    out["attribute_unit_mismatch"] = dict(unit_mismatch)
    return out


def _q(s: pd.Series, p: float) -> float | None:
    s = s.dropna()
    return None if s.empty else float(s.quantile(p))


def render_report(spec: dict, result: TaskResult, provenance: Mapping[str, Any]) -> str:
    s, f = result.summary, result.funnel
    L = [f"# Task preparation report: {spec['task']} (v{spec['version']})", "",
         "Analytical preparation only: no model was fitted. Observations were not modified, interpolated, resampled "
         "or imputed. Aggregate counts only.", "",
         f"- M1 input: `{provenance['m1_dir']}` (M1 manifest sha256 `{provenance['m1_manifest_sha256']}`)",
         f"- Task config sha256: `{provenance['task_config_sha256']}`; code: `{provenance['git_commit']}`",
         f"- Sampling-origin policy: `{spec['sampling_origin']['policy']}`", "",
         "## Sampling origin and interpretation", "", spec["sampling_origin"]["note"].strip(), "",
         "## Eligibility (per device)", "", "| Device | Step | Criterion | Excluded | Not evaluable | Remaining |",
         "|---|---|---|---|---|---|"]
    for r in f.itertuples():
        L.append(f"| {r.stream_id} | {r.step} | {r.criterion} | {r.excluded} | {r.not_evaluable} | {r.remaining} |")
    L += ["", f"Separation: no other meal in [t0 − {spec['separation']['before_min']} min, t0) or [t0, t0 + "
          f"{spec['separation']['after_min']} min]. Meals within {spec['near_duplicate_flag_min']} min of another "
          "meal are flagged as potentially duplicated or ambiguous and never merged.", "",
          "| Device | Candidates | Eligible | Candidates flagged near-duplicate |", "|---|---|---|---|"]
    for dev, d in s["devices"].items():
        L.append(f"| {dev} | {d['candidates']} | {d['eligible']} | {d['candidates_near_duplicate_flagged']} |")
    L += ["", "## Outcome calculability (eligible breakfasts)", "",
          "| Device | Calculable | Not calculable | Not computed | Reasons |", "|---|---|---|---|---|"]
    for dev, d in s["devices"].items():
        os_ = d["outcome_status"]
        reasons = ", ".join(f"{k}: {v}" for k, v in d["outcome_reasons"].items()) or "none"
        L.append(f"| {dev} | {os_['calculable']} | {os_['not_calculable']} | {os_['not_computed']} | {reasons} |")
    undecided = [f"{blk}.{key}[{dev}]" for blk, key in (("baseline", "min_readings"), ("outcome", "max_gap_min"))
                 for dev, v in spec[blk][key].items() if v is None]
    if undecided:
        L += ["", f"**Outcomes not computed:** undecided parameter(s) {', '.join(undecided)}. No default is applied."]
    L += ["", "Outcome parameters: minimum baseline readings "
          + ", ".join(f"{d} {v}" for d, v in spec["baseline"]["min_readings"].items())
          + "; longest interval the trapezoid may span (min) "
          + ", ".join(f"{d} {v}" for d, v in spec["outcome"]["max_gap_min"].items())
          + ". These are operational thresholds, not clinically validated. They apply to the readings as released "
          "(sampling origin as reported above), not to native device samples, so they cannot detect gaps that were "
          "filled before release."]
    calc = {d: v["iauc_signed_mgdl_min"] for d, v in s["devices"].items() if "iauc_signed_mgdl_min" in v}
    if calc:
        L += ["", "## Signed incremental AUC, 0–120 min (mg/dL × min; calculable outcomes, per device, not pooled)", "",
              "| Device | n | Mean (SD) | Median [IQR] | Min – max | Negative (below baseline) |", "|---|---|---|---|---|---|"]
        for dev, v in calc.items():
            sd = "n/a" if v["sd"] is None else f"{v['sd']:,.0f}"
            L.append(f"| {dev} | {v['n']} | {v['mean']:,.0f} ({sd}) | {v['median']:,.0f} [{v['p25']:,.0f} – "
                     f"{v['p75']:,.0f}] | {v['min']:,.0f} – {v['max']:,.0f} | {v['n_negative']} |")
        L += ["", "Signed: area below the baseline counts negatively and is not clamped. Descriptive only; no "
              "comparison between devices is made here."]
    L += ["", "Descriptive window characteristics of eligible breakfasts:", "",
          "| Device | Baseline readings (median / min) | Longest interval in outcome window, s (median / max) | Reading at window start | Reading at window end |",
          "|---|---|---|---|---|"]
    for dev, d in s["devices"].items():
        w = d["descriptive_eligible_windows"]
        L.append(f"| {dev} | {w['baseline_readings_median']} / {w['baseline_readings_min']} | "
                 f"{w['outcome_max_interval_s_median']} / {w['outcome_max_interval_s_max']} | "
                 f"{w['with_reading_at_window_start']} | {w['with_reading_at_window_end']} |")
    L += ["", "## Readings by sampling origin (eligible windows)", "", "| Device | Native | Interpolated | Unknown |",
          "|---|---|---|---|"]
    for dev, d in s["devices"].items():
        o = d["window_reading_origin"]
        L.append(f"| {dev} | {o['native']} | {o['interpolated']} | {o['unknown']} |")
    L += ["", "## Predictors", "", "Event predictors: " + ", ".join(spec["predictors"]["event"])
          + "; baseline glucose; participant attributes: "
          + ", ".join(f"{k} [{u}]" for k, u in spec["predictors"]["attributes"].items())
          + ". Missing predictor values are left missing (no imputation).", "",
          "| Device | " + " | ".join(next(iter(s["devices"].values()))["predictor_missing"]) + " |",
          "|---|" + "---|" * len(next(iter(s["devices"].values()))["predictor_missing"])]
    for dev, d in s["devices"].items():
        L.append(f"| {dev} | " + " | ".join(str(v) for v in d["predictor_missing"].values()) + " |")
    L += ["", f"Attribute values in a unit other than the requested one (not used): {s['attribute_unit_mismatch']}.", ""]
    return "\n".join(L)


def run(task_path: str | Path, m1_dir: str | Path, out_dir: str | Path, base_dir: str | Path) -> TaskResult:
    """Run the task on a finished M1 run (read-only) and write outputs, a manifest and an aggregate report."""
    spec = load_task(task_path)
    m1, out = Path(m1_dir), Path(out_dir)
    inputs = {n: m1 / f"{n}.parquet" for n in ("observations", "events", "participant_attributes")}
    inputs["m1_manifest"] = m1 / "_provenance" / "manifest.json"
    for p in inputs.values():
        if not p.exists():
            raise FileNotFoundError(f"M1 output not found: {p}")
    result = run_task(spec, pd.read_parquet(inputs["observations"]), pd.read_parquet(inputs["events"]),
                      pd.read_parquet(inputs["participant_attributes"]))
    out.mkdir(parents=True, exist_ok=True)
    manifest = RunManifest(f"task:{spec['task']}", base_dir)
    for p in [*inputs.values(), Path(task_path)]:
        manifest.add_input(p)
    profile_manifest = m1 / "quality" / "profile_manifest.json"
    if profile_manifest.exists():
        manifest.add_input(profile_manifest)
    manifest.add_config("task", spec)
    manifest.set_source({"task": spec["task"], "version": spec["version"], "dataset": spec["dataset"]})
    for name, df in (("ledger", result.ledger), ("funnel", result.funnel), ("windows", result.windows),
                     ("analysis_table", result.analysis_table)):
        path = out / f"{name}.parquet"
        df.to_parquet(path, index=False)
        manifest.add_output(path, table=name, rows=len(df))
    (out / "task_summary.json").write_text(json.dumps(result.summary, indent=2, sort_keys=True), encoding="utf-8")
    manifest.add_output(out / "task_summary.json")
    m1_sha = next(i["sha256"] for i in manifest.data["inputs"] if i["path"].endswith("_provenance/manifest.json"))
    report = render_report(spec, result, {"m1_dir": manifest._rel(m1), "m1_manifest_sha256": m1_sha,
                                          "task_config_sha256": config_hash(spec),
                                          "git_commit": manifest.data["git_commit"]})
    (out / "task_report.md").write_text(report, encoding="utf-8")
    manifest.add_output(out / "task_report.md")
    manifest.write(out / "task_manifest.json")
    return result
