"""Breakfast iAUC prediction experiment (M4): a bounded predictive research demonstration, not a framework.

Owner decisions P-1..P-8 (2026-11-25), implemented here:

* **Input:** the M3 breakfast analysis table, ``calculable`` rows only. The table's hash must match the M3
  task manifest, and the per-device calculable counts must match the M3 task summary.
* **Target:** the signed iAUC exactly as computed in M3 (negatives kept, nothing clamped or excluded).
* **Predictors:** the approved set, unchanged (:data:`APPROVED_PREDICTORS`). An analysis may only *drop*
  predictors (sensitivity analysis); outcome-window and identifier columns can never be used.
* **Devices:** every step runs independently per device; nothing is pooled and no device is primary.
* **Validation:** leave-one-participant-out (all breakfasts of a participant stay together). No hold-out set,
  no nested CV, no tuning: model settings are fixed in the config. This is internal participant-level
  validation within one dataset, not external validation.
* **Models:** naive (training-fold mean), ordinary least squares (numpy), and a random forest with fixed
  settings (scikit-learn, optional; skipped with a recorded reason when unavailable).
* **Primary metric:** participant-macro MAE (MAE per held-out participant, averaged over participants).
  Pooled breakfast-level MAE, RMSE, R² against the naive model and calibration are secondary.
* **Uncertainty:** percentile intervals from a bootstrap over whole participants; paired comparisons use the
  same resamples for both models. The forest is an improvement over linear regression only if the
  interval for (linear − forest) macro MAE excludes zero on the positive side.

Run: ``python -m pghd_ready.conditions.cardiometabolic.experiments.breakfast_iauc --config ... --task-dir ...
--out ...``. Participant-level outputs (split assignments, predictions) must stay in git-ignored locations;
the report and metrics are aggregate.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from dataclasses import dataclass, field
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ....core.provenance import RunManifest, config_hash, sha256_file

TARGET = "iauc_signed_mgdl_min"
GROUP = "participant_id"
DEVICE = "stream_id"
APPROVED_PREDICTORS = ("energy_kcal", "carbohydrate_g", "protein_g", "fat_g", "fiber_g",
                       "baseline_glucose_mgdl", "hba1c", "body_mass_index", "age")
MODELS = ("naive", "linear", "random_forest")
SUBSETS = ("all_calculable", "common_breakfasts")
# (improved model, reference model): the difference is reference MAE − improved MAE; positive = improvement.
COMPARISONS = (("linear", "naive"), ("random_forest", "naive"), ("random_forest", "linear"))
RF_UNAVAILABLE = ("scikit-learn is not installed (optional extra 'experiment'); the random forest was skipped. "
                  "The naive and linear models do not need it.")


# ------------------------------------------------------------------ configuration
def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict) -> None:
    if cfg.get("target") != TARGET:
        raise ValueError(f"target must be {TARGET!r} (signed iAUC from M3), got {cfg.get('target')!r}")
    preds = list(cfg.get("predictors") or [])
    if sorted(preds) != sorted(APPROVED_PREDICTORS) or len(set(preds)) != len(preds):
        raise ValueError("predictors must be exactly the approved set "
                         f"{list(APPROVED_PREDICTORS)}; got {preds}")
    devices = list(cfg.get("devices") or [])
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("devices must be a non-empty list without duplicates")
    val = cfg.get("validation") or {}
    if val.get("scheme") != "leave_one_participant_out" or val.get("group") != GROUP:
        raise ValueError("validation must be leave_one_participant_out grouped by participant_id")
    models = cfg.get("models") or {}
    if set(models) != set(MODELS):
        raise ValueError(f"models must be exactly {list(MODELS)}")
    analyses = cfg.get("analyses") or []
    names = [a.get("name") for a in analyses]
    if "primary" not in names or len(set(names)) != len(names):
        raise ValueError("analyses need unique names, including one named 'primary'")
    for a in analyses:
        if a.get("subset") not in SUBSETS:
            raise ValueError(f"analysis {a['name']!r}: subset must be one of {SUBSETS}")
        drop = list(a.get("drop_predictors") or [])
        if not set(drop) <= set(preds) or len(drop) >= len(preds):
            raise ValueError(f"analysis {a['name']!r}: drop_predictors must be a proper subset of the predictors")
        if a["name"] == "primary" and (drop or a["subset"] != "all_calculable"):
            raise ValueError("the primary analysis uses all calculable breakfasts and every predictor")
    boot = cfg.get("bootstrap") or {}
    if not (int(boot.get("resamples", 0)) > 0 and 0 < float(boot.get("level", 0)) < 1 and "seed" in boot):
        raise ValueError("bootstrap needs resamples > 0, 0 < level < 1 and a seed")


def analysis_predictors(cfg: dict, analysis: dict) -> list[str]:
    drop = set(analysis.get("drop_predictors") or [])
    return [p for p in cfg["predictors"] if p not in drop]


# ------------------------------------------------------------------ data selection
def select_rows(table: pd.DataFrame, devices: list[str], subset: str) -> dict[str, pd.DataFrame]:
    """Calculable rows per device (copies, sorted), optionally restricted to breakfasts calculable on every device."""
    missing_cols = {TARGET, GROUP, DEVICE, "event_id", "outcome_status", *APPROVED_PREDICTORS} - set(table.columns)
    if missing_cols:
        raise ValueError(f"analysis table lacks columns: {sorted(missing_cols)}")
    calc = table[table["outcome_status"] == "calculable"]
    absent = [d for d in devices if d not in set(calc[DEVICE])]
    if absent:
        raise ValueError(f"no calculable rows for configured devices: {absent}")
    if subset == "common_breakfasts":
        per_event = calc[calc[DEVICE].isin(devices)].groupby("event_id")[DEVICE].nunique()
        calc = calc[calc["event_id"].isin(per_event.index[per_event == len(devices)])]
    out = {}
    for d in devices:
        df = calc[calc[DEVICE] == d].sort_values([GROUP, "event_id"]).reset_index(drop=True)
        if df["event_id"].duplicated().any():
            raise ValueError(f"{d}: duplicate event_id in calculable rows")
        na = df[[TARGET, *APPROVED_PREDICTORS]].isna().sum()
        if na.any():
            raise ValueError(f"{d}: missing values (no imputation is performed): {na[na > 0].to_dict()}")
        out[d] = df
    return out


# ------------------------------------------------------------------ models
def _ols_fit(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    beta, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(X)), X]), y, rcond=None)
    return beta


def _ols_predict(beta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return beta[0] + X @ beta[1:]


def random_forest_available() -> bool:
    try:
        import sklearn.ensemble  # noqa: F401
    except ImportError:
        return False
    return True


def _forest_fit(settings: dict, X: np.ndarray, y: np.ndarray):
    """Fit with all cores (trees are seeded, so fitting is deterministic), then predict single-threaded:
    multi-threaded prediction sums the trees in arbitrary order, which changes the last floating-point digits."""
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(n_jobs=-1, **settings).fit(X, y).set_params(n_jobs=1)


def lopo_predictions(df: pd.DataFrame, predictors: list[str], cfg: dict, use_forest: bool) -> pd.DataFrame:
    """Leave-one-participant-out out-of-fold predictions. Fold k holds out the k-th participant (sorted)."""
    X, y, g = df[predictors].to_numpy(float), df[TARGET].to_numpy(float), df[GROUP].to_numpy()
    models = [m for m in MODELS if m != "random_forest" or use_forest]
    pred = {m: np.full(len(y), np.nan) for m in models}
    fold = np.full(len(y), -1)
    for k, p in enumerate(np.unique(g)):
        test = g == p
        train = ~test
        if set(g[train]) & set(g[test]) or not train.any():
            raise AssertionError("participant leakage between training and test partitions")
        fold[test] = k
        pred["naive"][test] = y[train].mean()
        pred["linear"][test] = _ols_predict(_ols_fit(X[train], y[train]), X[test])
        if use_forest:
            pred["random_forest"][test] = _forest_fit(cfg["models"]["random_forest"], X[train], y[train]).predict(X[test])
    out = pd.DataFrame({GROUP: g, "event_id": df["event_id"].to_numpy(), "fold": fold, "observed": y})
    for m in models:
        out[f"pred_{m}"] = pred[m]
    return out


# ------------------------------------------------------------------ metrics
def _participant_errors(oof: pd.DataFrame, models: list[str]) -> pd.DataFrame:
    e = pd.DataFrame({GROUP: oof[GROUP]})
    for m in models:
        r = oof["observed"] - oof[f"pred_{m}"]
        e[f"ae_{m}"], e[f"se_{m}"] = r.abs(), r ** 2
    agg = e.groupby(GROUP, sort=True).agg(n=(GROUP, "size"), **{f"sae_{m}": (f"ae_{m}", "sum") for m in models},
                                          **{f"sse_{m}": (f"se_{m}", "sum") for m in models})
    return agg


def point_metrics(oof: pd.DataFrame, model: str) -> dict[str, Any]:
    obs, pr = oof["observed"].to_numpy(), oof[f"pred_{model}"].to_numpy()
    pp_mae = (oof["observed"] - oof[f"pred_{model}"]).abs().groupby(oof[GROUP]).mean()
    sse, sse_naive = float(((obs - pr) ** 2).sum()), float(((obs - oof["pred_naive"]) ** 2).sum())
    m = {"participant_macro_mae": float(pp_mae.mean()), "pooled_mae": float(np.abs(obs - pr).mean()),
         "rmse": float(np.sqrt(((obs - pr) ** 2).mean())),
         "participant_mae_median": float(pp_mae.median()), "participant_mae_min": float(pp_mae.min()),
         "participant_mae_max": float(pp_mae.max()), "r2_vs_naive": None,
         "calibration_slope": None, "calibration_intercept": None}
    if model != "naive":
        m["r2_vs_naive"] = 1.0 - sse / sse_naive
        slope, intercept = np.polyfit(pr, obs, 1)       # observed regressed on predicted
        m["calibration_slope"], m["calibration_intercept"] = float(slope), float(intercept)
    return m


def bootstrap_weights(n_participants: int, resamples: int, seed_entropy: list[int]) -> np.ndarray:
    """(resamples x participants) multiplicities of whole participants drawn with replacement."""
    idx = np.random.default_rng(seed_entropy).integers(0, n_participants, size=(resamples, n_participants))
    return np.stack([np.bincount(r, minlength=n_participants) for r in idx]).astype(float)


def bootstrap_statistics(agg: pd.DataFrame, models: list[str], W: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    n = agg["n"].to_numpy(float)
    out = {}
    for m in models:
        sae, sse = agg[f"sae_{m}"].to_numpy(), agg[f"sse_{m}"].to_numpy()
        out[m] = {"participant_macro_mae": W @ (sae / n) / W.sum(axis=1), "pooled_mae": (W @ sae) / (W @ n),
                  "rmse": np.sqrt((W @ sse) / (W @ n))}
        if m != "naive":
            out[m]["r2_vs_naive"] = 1.0 - (W @ sse) / (W @ agg["sse_naive"].to_numpy())
    return out


def _interval(x: np.ndarray, level: float) -> list[float]:
    a = (1 - level) / 2
    return [float(np.quantile(x, a)), float(np.quantile(x, 1 - a))]


def _difference(ref: np.ndarray, new: np.ndarray, point: float, level: float) -> dict[str, Any]:
    lo, hi = _interval(ref - new, level)
    return {"point": point, "interval": [lo, hi], "excludes_zero": bool(lo > 0 or hi < 0), "improvement": bool(lo > 0)}


# ------------------------------------------------------------------ experiment
@dataclass
class ExperimentResult:
    metrics: dict
    model_metadata: dict
    oof_predictions: pd.DataFrame
    split_assignments: pd.DataFrame
    boot: dict = field(default_factory=dict, repr=False)


def _vif(X: np.ndarray) -> list[float]:
    out = []
    for j in range(X.shape[1]):
        others = np.delete(X, j, axis=1)
        fit = _ols_predict(_ols_fit(others, X[:, j]), others)
        r2 = 1 - ((X[:, j] - fit) ** 2).sum() / ((X[:, j] - X[:, j].mean()) ** 2).sum()
        out.append(float(1 / (1 - r2)) if r2 < 1 else float("inf"))
    return out


def run_experiment(cfg: dict, table: pd.DataFrame, use_forest: bool | None = None) -> ExperimentResult:
    validate_config(cfg)
    if use_forest is None:
        use_forest = random_forest_available()
    models = [m for m in MODELS if m != "random_forest" or use_forest]
    boot_cfg, level = cfg["bootstrap"], float(cfg["bootstrap"]["level"])
    metrics: dict[str, Any] = {"random_forest": {"run": use_forest, "reason": None if use_forest else RF_UNAVAILABLE},
                               "primary_metric": "participant_macro_mae", "level": level, "analyses": {}}
    meta: dict[str, Any] = {"models": {"naive": "training-fold mean of the target",
                                       "linear": "ordinary least squares with intercept (numpy.linalg.lstsq)",
                                       "random_forest": dict(cfg["models"]["random_forest"], n_jobs_fit=-1, n_jobs_predict=1)
                                       if use_forest else None},
                            "fits": {}}
    oofs, splits, boot = [], [], {}
    for a in cfg["analyses"]:
        preds = analysis_predictors(cfg, a)
        rows = select_rows(table, cfg["devices"], a["subset"])
        metrics["analyses"][a["name"]] = {}
        meta["fits"][a["name"]] = {}
        for d_idx, d in enumerate(cfg["devices"]):
            df = rows[d]
            oof = lopo_predictions(df, preds, cfg, use_forest)
            agg = _participant_errors(oof, models)
            # Same seed per device for every analysis: analyses over the same participants share resamples (paired).
            W = bootstrap_weights(len(agg), int(boot_cfg["resamples"]), [int(boot_cfg["seed"]), d_idx])
            bs = bootstrap_statistics(agg, models, W)
            boot[(a["name"], d)] = (tuple(agg.index), bs)
            res = {"n_breakfasts": len(df), "n_participants": int(df[GROUP].nunique()), "predictors": preds,
                   "subset": a["subset"], "models": {}, "comparisons": {}}
            for m in models:
                pm = point_metrics(oof, m)
                pm["intervals"] = {k: _interval(v, level) for k, v in bs[m].items()}
                res["models"][m] = pm
            for new, ref in COMPARISONS:
                if new in models and ref in models:
                    key = f"{new}_vs_{ref}"
                    res["comparisons"][key] = {
                        s: _difference(bs[ref][s], bs[new][s],
                                       res["models"][ref][s] - res["models"][new][s], level)
                        for s in ("participant_macro_mae", "pooled_mae")}
            rf_cmp = res["comparisons"].get("random_forest_vs_linear")
            res["random_forest_improves_on_linear"] = None if rf_cmp is None else rf_cmp["participant_macro_mae"]["improvement"]
            metrics["analyses"][a["name"]][d] = res
            oofs.append(oof.assign(analysis=a["name"], device=d))
            splits.append(oof.groupby([GROUP, "fold"], as_index=False).size().rename(columns={"size": "n_test_breakfasts"})
                          .assign(analysis=a["name"], device=d))
            X, y = df[preds].to_numpy(float), df[TARGET].to_numpy(float)
            beta = _ols_fit(X, y)
            fit = {"rows": len(df), "linear_coefficients": {"intercept": float(beta[0]),
                                                            **{p: float(b) for p, b in zip(preds, beta[1:])}},
                   "variance_inflation_factor": dict(zip(preds, _vif(X)))}
            if use_forest:
                rf = _forest_fit(cfg["models"]["random_forest"], X, y)
                fit["random_forest_impurity_importance"] = {p: float(v) for p, v in zip(preds, rf.feature_importances_)}
            meta["fits"][a["name"]][d] = fit
    metrics["cross_analysis"] = _cross_analysis(cfg, boot, models, metrics, level)
    cols = ["analysis", "device", GROUP, "event_id", "fold", "observed", *[f"pred_{m}" for m in models]]
    oof_all = pd.concat(oofs, ignore_index=True)[cols]
    split_all = pd.concat(splits, ignore_index=True)[["analysis", "device", GROUP, "fold", "n_test_breakfasts"]]
    return ExperimentResult(metrics, meta, oof_all, split_all, boot)


def _cross_analysis(cfg, boot, models, metrics, level) -> dict:
    """Paired (same participants, same resamples) macro-MAE difference of each sensitivity analysis vs primary."""
    out = {}
    for a in cfg["analyses"]:
        if a["name"] == "primary":
            continue
        out[a["name"]] = {}
        for d in cfg["devices"]:
            p_ids, p_bs = boot[("primary", d)]
            s_ids, s_bs = boot[(a["name"], d)]
            if p_ids != s_ids:
                out[a["name"]][d] = {"paired": False, "reason": "different participants from the primary analysis"}
                continue
            diffs = {}
            for m in models:
                if m == "naive":
                    continue
                dd = _difference(s_bs[m]["participant_macro_mae"], p_bs[m]["participant_macro_mae"],
                                 metrics["analyses"][a["name"]][d]["models"][m]["participant_macro_mae"]
                                 - metrics["analyses"]["primary"][d]["models"][m]["participant_macro_mae"], level)
                dd.pop("improvement")   # here the difference is sensitivity − primary: positive means worse
                diffs[m] = dd
            out[a["name"]][d] = {"paired": True, "difference": "sensitivity minus primary macro MAE", **diffs}
    return out


# ------------------------------------------------------------------ report
LABEL = {"naive": "Naive (training mean)", "linear": "Linear regression", "random_forest": "Random forest"}
DEVICE_LABEL = {"libre_pro": "Libre Pro", "dexcom_g6": "Dexcom G6"}


def _n(x: float | None, nd: int = 0) -> str:
    return "–" if x is None else f"{x:,.{nd}f}".replace("-", "−")


def _ci(v: float, iv: list[float], nd: int = 0) -> str:
    return f"{_n(v, nd)} [{_n(iv[0], nd)}, {_n(iv[1], nd)}]"


def render_report(cfg: dict, res: ExperimentResult, prov: dict) -> str:
    M, devs = res.metrics, cfg["devices"]
    models = [m for m in MODELS if m in M["analyses"]["primary"][devs[0]]["models"]]
    pct = f"{M['level']:.0%}"
    head = " | ".join(DEVICE_LABEL.get(d, d) for d in devs)
    sep = "|---" * (len(devs) + 1) + "|"
    L = ["# Breakfast iAUC prediction experiment (M4)", "",
         f"- Experiment `{cfg['experiment']}` v{cfg['version']}; config sha256 `{prov['config_sha256']}`; "
         f"code `{prov['git_commit']}`",
         f"- Input: M3 analysis table sha256 `{prov['analysis_table_sha256']}` (matches the M3 task manifest); "
         f"M1 manifest sha256 `{prov['m1_manifest_sha256']}`",
         f"- Validation: leave-one-participant-out; {pct} intervals from {cfg['bootstrap']['resamples']:,} "
         "bootstrap resamples of whole participants (paired across models).", "",
         "## Scope and limitations", "",
         "- **Internal validation only.** Leave-one-participant-out estimates performance for held-out participants "
         "*within CGMacros*. It is not external validation and is not evidence of clinical generalisability.",
         "- **The target is dataset-derived.** It is the signed iAUC (0–120 min) calculated in M3 from CGM values as "
         "released, which the dataset authors document as linearly interpolated to one-minute rows. It is not a "
         "measured physiological glucose response. Negative values are kept (signed iAUC; area below baseline is not clamped).",
         "- **Removing baseline glucose as a predictor does not remove interpolation-related contamination of the "
         "target itself.** The no-baseline-glucose analysis only tests the predictor side.",
         "- **Devices are analysed independently and never pooled.** They share participants, so their results are "
         "not independent replications, and their calculable breakfasts differ. No device is primary.",
         "- Predictors are the approved set only. Coefficients and importances in `model_metadata.json` are "
         "descriptive model parameters, not effects; no clinical interpretation is offered.", ""]
    if not M["random_forest"]["run"]:
        L += [f"> **Random forest not run:** {M['random_forest']['reason']}", ""]
    L += ["## Data", "", f"| Analysis | {head} |", sep]
    for a in cfg["analyses"]:
        r = M["analyses"][a["name"]]
        L.append(f"| {a['name']} | " + " | ".join(
            f"{r[d]['n_breakfasts']} breakfasts, {r[d]['n_participants']} participants" for d in devs) + " |")
    for a in cfg["analyses"]:
        r = M["analyses"][a["name"]]
        title = {"primary": "Primary analysis",
                 "no_baseline_glucose": "Sensitivity: without baseline glucose as a predictor",
                 "common_breakfasts": "Sensitivity: breakfasts calculable on every device"}.get(a["name"], a["name"])
        L += ["", f"## {title}", "",
              f"Participant-macro MAE (primary metric; mg/dL × min), point estimate [{pct} interval]. "
              f"Predictors: {', '.join(r[devs[0]]['predictors'])}.", "",
              f"| Model | {head} |", sep]
        for m in models:
            L.append(f"| {LABEL[m]} | " + " | ".join(
                _ci(r[d]["models"][m]["participant_macro_mae"], r[d]["models"][m]["intervals"]["participant_macro_mae"])
                for d in devs) + " |")
        L += ["", f"Paired macro-MAE reduction (reference − model; positive = model better) [{pct} interval]:", "",
              f"| Comparison | {head} |", sep]
        for key in r[devs[0]]["comparisons"]:
            new, ref = key.split("_vs_")
            L.append(f"| {LABEL[new]} vs {LABEL[ref].split(' (')[0].lower()} | " + " | ".join(
                _ci(r[d]["comparisons"][key]["participant_macro_mae"]["point"],
                    r[d]["comparisons"][key]["participant_macro_mae"]["interval"])
                + (" (excludes 0)" if r[d]["comparisons"][key]["participant_macro_mae"]["excludes_zero"] else "")
                for d in devs) + " |")
        if "random_forest" in models:
            L += ["", "Random forest an improvement over linear regression (P-7 rule: interval excludes zero, "
                  "positive side): " + "; ".join(
                      f"{DEVICE_LABEL.get(d, d)} **{'yes' if r[d]['random_forest_improves_on_linear'] else 'no'}**"
                      for d in devs) + "."]
        if a["name"] in M["cross_analysis"]:
            ca = M["cross_analysis"][a["name"]]
            L += ["", f"Change in macro MAE versus the primary analysis (this analysis − primary; positive = worse) "
                  f"[{pct} interval]:", "", f"| Model | {head} |", sep]
            for m in models:
                if m == "naive":
                    continue
                L.append(f"| {LABEL[m]} | " + " | ".join(
                    _ci(ca[d][m]["point"], ca[d][m]["interval"]) if ca[d].get("paired")
                    else "not paired (different participants)" for d in devs) + " |")
            if a["name"] == "no_baseline_glucose":
                L += ["", "This tests only whether the baseline-glucose predictor drives performance. It does not "
                      "remove potential interpolation-related contamination in the iAUC target."]
    p = M["analyses"]["primary"]
    L += ["", "## Secondary metrics (primary analysis)", "",
          f"| Model | Device | Pooled MAE [{pct}] | RMSE [{pct}] | R² vs naive [{pct}] | Calibration slope "
          "| Per-participant MAE median (min–max) |", "|---|---|---|---|---|---|---|"]
    for m in models:
        for d in devs:
            x = p[d]["models"][m]
            r2 = "–" if x["r2_vs_naive"] is None else _ci(x["r2_vs_naive"], x["intervals"]["r2_vs_naive"], 3)
            L.append(f"| {LABEL[m]} | {DEVICE_LABEL.get(d, d)} | {_ci(x['pooled_mae'], x['intervals']['pooled_mae'])} "
                     f"| {_ci(x['rmse'], x['intervals']['rmse'])} | {r2} | {_n(x['calibration_slope'], 2)} "
                     f"| {_n(x['participant_mae_median'])} ({_n(x['participant_mae_min'])}–{_n(x['participant_mae_max'])}) |")
    vif = {d: max(res.model_metadata["fits"]["primary"][d]["variance_inflation_factor"].items(), key=lambda kv: kv[1])
           for d in devs}
    L += ["", "## Notes", "",
          "- Collinearity (primary analysis, all calculable breakfasts): largest variance inflation factor "
          + "; ".join(f"{DEVICE_LABEL.get(d, d)} {_n(v, 1)} ({k})" for d, (k, v) in vif.items())
          + ". Energy is close to a linear combination of the macronutrients, so coefficients are not interpretable "
          "as separate effects.",
          "- Participant characteristics (HbA1c, BMI, age) take one value per participant, so they rest on "
          f"{p[devs[0]]['n_participants']} independent units, not on the number of breakfasts.",
          "- Participant-level files (`split_assignments.parquet`, `oof_predictions.parquet`) must stay local.", ""]
    return "\n".join(L)


# ------------------------------------------------------------------ run
def _versions() -> dict[str, str | None]:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "pandas", "scikit-learn"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            out[pkg] = None
    return out


def run(config_path: str | Path, task_dir: str | Path, out_dir: str | Path, base_dir: str | Path,
        use_forest: bool | None = None) -> ExperimentResult:
    cfg = load_config(config_path)
    task, out = Path(task_dir), Path(out_dir)
    paths = {n: task / n for n in ("analysis_table.parquet", "task_manifest.json", "task_summary.json")}
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(f"M3 output not found: {p}")
    task_manifest = json.loads(paths["task_manifest.json"].read_text(encoding="utf-8"))
    table_sha = sha256_file(paths["analysis_table.parquet"])
    recorded = [o["sha256"] for o in task_manifest["outputs"] if o["path"].endswith("analysis_table.parquet")]
    if recorded != [table_sha]:
        raise ValueError("analysis table hash does not match the M3 task manifest (lineage broken)")
    if task_manifest.get("source", {}).get("task") != cfg["input_task"]:
        raise ValueError(f"M3 task is {task_manifest.get('source', {}).get('task')!r}, config expects {cfg['input_task']!r}")
    m1_sha = next(i["sha256"] for i in task_manifest["inputs"] if i["path"].endswith("_provenance/manifest.json"))
    summary = json.loads(paths["task_summary.json"].read_text(encoding="utf-8"))
    table = pd.read_parquet(paths["analysis_table.parquet"])
    for d in cfg["devices"]:
        n = int(((table[DEVICE] == d) & (table["outcome_status"] == "calculable")).sum())
        if n != summary["devices"][d]["outcome_status"]["calculable"]:
            raise ValueError(f"{d}: {n} calculable rows, but the M3 summary records "
                             f"{summary['devices'][d]['outcome_status']['calculable']}")
    res = run_experiment(cfg, table, use_forest)

    out.mkdir(parents=True, exist_ok=True)
    manifest = RunManifest(f"experiment:{cfg['experiment']}", base_dir)
    for p in [*paths.values(), Path(config_path)]:
        manifest.add_input(p)
    manifest.add_config("experiment", cfg)
    manifest.set_source({"experiment": cfg["experiment"], "version": cfg["version"], "input_task": cfg["input_task"]})
    manifest.data["environment"] = _versions()
    manifest.data["lineage"] = {
        "m1_manifest_sha256": m1_sha, "analysis_table_sha256": table_sha,
        "row_selection": "outcome_status == 'calculable', one device at a time; subset per analysis",
        "column_selection": {a["name"]: analysis_predictors(cfg, a) for a in cfg["analyses"]},
        "transformations": "none (no scaling, imputation, encoding or outcome transformation)"}
    for name, df in (("split_assignments", res.split_assignments), ("oof_predictions", res.oof_predictions)):
        path = out / f"{name}.parquet"
        df.to_parquet(path, index=False)
        manifest.add_output(path, table=name, rows=len(df))
    meta = dict(res.model_metadata, predictors_in_order=cfg["predictors"], environment=_versions(),
                bootstrap=cfg["bootstrap"], validation=cfg["validation"])
    for name, obj in (("metrics.json", res.metrics), ("model_metadata.json", meta)):
        (out / name).write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
        manifest.add_output(out / name)
    report = render_report(cfg, res, {"config_sha256": config_hash(cfg), "git_commit": manifest.data["git_commit"],
                                      "analysis_table_sha256": table_sha, "m1_manifest_sha256": m1_sha})
    (out / "experiment_report.md").write_text(report, encoding="utf-8")
    manifest.add_output(out / "experiment_report.md")
    manifest.write(out / "experiment_manifest.json")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pghd_ready.conditions.cardiometabolic.experiments.breakfast_iauc",
                                 description="M4 breakfast iAUC prediction experiment (internal LOPO validation).")
    ap.add_argument("--config", required=True, type=Path, help="experiment config (configs/experiments/...)")
    ap.add_argument("--task-dir", required=True, type=Path, help="M3 task output directory")
    ap.add_argument("--out", required=True, type=Path, help="output directory (must be git-ignored)")
    ap.add_argument("--base-dir", type=Path, default=Path.cwd(), help="base for relative paths in the manifest")
    args = ap.parse_args(argv)
    try:
        res = run(args.config, args.task_dir, args.out, args.base_dir)
    except Exception as exc:
        print(f"Experiment FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    if not res.metrics["random_forest"]["run"]:
        print(f"Note: {RF_UNAVAILABLE}", file=sys.stderr)
    print(f"Experiment SUCCESS: {len(res.oof_predictions)} out-of-fold predictions")
    print(f"Report: {args.out / 'experiment_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
