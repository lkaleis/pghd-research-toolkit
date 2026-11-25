"""Breakfast iAUC model refinement (M4): collinearity, fasting glucose and ridge regularisation.

Owner approval 2026-11-25. A separate experiment beside the M4 reference (:mod:`.breakfast_iauc`), whose module,
config and outputs are only read:

* **Rows:** exactly the M4 primary rows (all calculable breakfasts per device). The run asserts that its
  reference model A and the naive model reproduce the M4 out-of-fold predictions.
* **Models** (per device, leave-one-participant-out): A = OLS with the nine approved predictors (M4 reference);
  B = A without total energy; C = B plus study-start fasting glucose; D1/D2 = ridge on B/C.
* **Fasting glucose** (``fasting_glucose_lab``, mg/dL) is joined from the M1 participant attributes. It is one
  laboratory value per participant documented as fasting and taken at study start, distinct from the
  per-breakfast baseline glucose; its collection date and relation to individual breakfasts cannot be verified.
* **Ridge:** predictors standardised with training-fold statistics only; intercept unpenalised; alpha chosen from a
  fixed grid by inner leave-one-participant-out over the outer training participants (scaling refitted inside
  every inner training fold), criterion participant-macro MAE, ties to the larger alpha. The held-out
  participant is never used for scaling or selection.
* **Comparisons:** paired improvement = MAE(reference) − MAE(candidate) (positive = candidate lower error), with
  the same whole-participant bootstrap resamples for every model (and the same seeds as M4).

Run: ``python -m pghd_ready.conditions.cardiometabolic.experiments.breakfast_iauc_refinement --config ...
--task-dir ... --m1 ... --reference-dir ... --out ...``. Participant-level outputs must stay local.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from ....core import alignment as al
from ....core.provenance import RunManifest, config_hash, sha256_file
from . import breakfast_iauc as m4

TARGET, GROUP, DEVICE = m4.TARGET, m4.GROUP, m4.DEVICE
ADDED = "fasting_glucose_lab"
ALLOWED_PREDICTORS = (*m4.APPROVED_PREDICTORS, ADDED)
REFERENCE_MODEL = "A_ols_full"
STAT_KEYS = ("participant_macro_mae", "pooled_mae", "rmse")


# ------------------------------------------------------------------ configuration
def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    validate_config(cfg)
    return cfg


def validate_config(cfg: dict, reference_cfg: dict | None = None) -> None:
    if cfg.get("target") != TARGET:
        raise ValueError(f"target must be {TARGET!r} (signed iAUC from M3)")
    if cfg.get("added_attributes") != {ADDED: "mg/dL"}:
        raise ValueError(f"the only added predictor allowed is {ADDED!r} in mg/dL")
    if (cfg.get("validation") or {}) != {"scheme": "leave_one_participant_out", "group": GROUP}:
        raise ValueError("validation must be leave_one_participant_out grouped by participant_id")
    models = cfg.get("models") or {}
    if REFERENCE_MODEL not in models or models[REFERENCE_MODEL] != {"type": "ols", "predictors": list(m4.APPROVED_PREDICTORS)}:
        raise ValueError(f"{REFERENCE_MODEL} must be OLS with the approved predictors in M4 order")
    for name, spec in models.items():
        preds = list(spec.get("predictors") or [])
        if spec.get("type") not in ("ols", "ridge"):
            raise ValueError(f"model {name!r}: type must be ols or ridge")
        if not preds or len(set(preds)) != len(preds) or not set(preds) <= set(ALLOWED_PREDICTORS):
            raise ValueError(f"model {name!r}: predictors must be distinct and from {list(ALLOWED_PREDICTORS)}")
    ridge = cfg.get("ridge") or {}
    alphas = list(ridge.get("alphas") or [])
    if (not alphas or any(a <= 0 for a in alphas) or alphas != sorted(set(alphas))
            or ridge.get("selection") != "inner_leave_one_participant_out"
            or ridge.get("criterion") != "participant_macro_mae" or ridge.get("ties") != "larger_alpha"):
        raise ValueError("ridge needs a sorted grid of positive alphas, inner LOPO selection on participant-macro MAE "
                         "and ties to the larger alpha")
    for c in cfg.get("comparisons") or []:
        if c.get("candidate") not in models or c.get("reference") not in models or c["candidate"] == c["reference"]:
            raise ValueError(f"invalid comparison {c}")
    if not cfg.get("comparisons"):
        raise ValueError("comparisons must not be empty")
    devices = list(cfg.get("devices") or [])
    if not devices or len(set(devices)) != len(devices):
        raise ValueError("devices must be a non-empty list without duplicates")
    if reference_cfg is not None:
        if reference_cfg["bootstrap"] != cfg.get("bootstrap"):
            raise ValueError("bootstrap settings must equal the M4 reference (same participant resamples)")
        if reference_cfg["devices"] != devices:
            raise ValueError("devices must be in the same order as the M4 reference (paired bootstrap seeds)")
        if cfg["reference"].get("analysis") != "primary":
            raise ValueError("the refinement uses the M4 primary rows only")


# ------------------------------------------------------------------ data
def add_attributes(rows: pd.DataFrame, attributes: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Join the configured participant attribute(s) in exactly the configured unit; refuse gaps or duplicates."""
    wide = al.attributes_wide(attributes, cfg["added_attributes"], dataset=cfg.get("dataset"))
    if any(wide.attrs["unit_mismatch"].values()):
        raise ValueError(f"attribute values in another unit: {wide.attrs['unit_mismatch']}")
    out = rows.merge(wide, on=GROUP, how="left", validate="many_to_one")
    missing = out[list(cfg["added_attributes"])].isna().sum()
    if missing.any():
        raise ValueError(f"added attribute missing for some breakfasts (no imputation): {missing[missing > 0].to_dict()}")
    return out


# ------------------------------------------------------------------ ridge (numpy; training-fold scaling only)
def ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float) -> dict[str, np.ndarray | float]:
    mu, sd = X.mean(axis=0), X.std(axis=0)
    if (sd == 0).any():
        raise ValueError("a predictor is constant in a training fold; it cannot be standardised")
    Z = (X - mu) / sd
    n = len(y)
    b = np.linalg.solve(Z.T @ Z / n + alpha * np.eye(Z.shape[1]), Z.T @ (y - y.mean()) / n)
    return {"mu": mu, "sd": sd, "beta_std": b, "intercept": float(y.mean())}


def ridge_predict(fit: dict, X: np.ndarray) -> np.ndarray:
    return fit["intercept"] + ((X - fit["mu"]) / fit["sd"]) @ fit["beta_std"]


def select_alpha(X: np.ndarray, y: np.ndarray, g: np.ndarray, alphas: list[float]) -> tuple[float, list[float]]:
    """Inner leave-one-participant-out over the given (training) participants only. Returns the alpha with the
    lowest inner participant-macro MAE (ties to the larger alpha) and the inner score for every alpha."""
    parts = np.unique(g)
    mae = np.zeros((len(parts), len(alphas)))
    for i, q in enumerate(parts):
        te, tr = g == q, g != q
        for j, a in enumerate(alphas):
            mae[i, j] = np.abs(y[te] - ridge_predict(ridge_fit(X[tr], y[tr], a), X[te])).mean()
    score = mae.mean(axis=0)
    best = np.flatnonzero(np.isclose(score, score.min(), rtol=1e-12, atol=0.0))
    return float(alphas[best.max()]), [float(s) for s in score]


def _ols_std_coefficients(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """OLS coefficients per training-fold SD of each predictor (same predictions as unstandardised OLS)."""
    sd = X.std(axis=0)
    return m4._ols_fit(X, y)[1:] * sd


# ------------------------------------------------------------------ experiment
@dataclass
class RefinementResult:
    metrics: dict
    model_metadata: dict
    oof_predictions: pd.DataFrame
    split_assignments: pd.DataFrame
    fold_coefficients: pd.DataFrame


def lopo(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    y, g = df[TARGET].to_numpy(float), df[GROUP].to_numpy()
    parts = np.unique(g)
    names = list(cfg["models"])
    pred = {"naive": np.full(len(y), np.nan), **{m: np.full(len(y), np.nan) for m in names}}
    fold = np.full(len(y), -1)
    coef_rows = []
    alphas = list(cfg["ridge"]["alphas"])
    for k, p in enumerate(parts):
        te, tr = g == p, g != p
        if set(g[tr]) & set(g[te]):
            raise AssertionError("participant leakage between training and test partitions")
        fold[te] = k
        pred["naive"][te] = y[tr].mean()
        for m, spec in cfg["models"].items():
            X = df[spec["predictors"]].to_numpy(float)
            if spec["type"] == "ols":
                pred[m][te] = m4._ols_predict(m4._ols_fit(X[tr], y[tr]), X[te])
                coefs, alpha = _ols_std_coefficients(X[tr], y[tr]), None
            else:
                alpha, _ = select_alpha(X[tr], y[tr], g[tr], alphas)
                fit = ridge_fit(X[tr], y[tr], alpha)
                pred[m][te] = ridge_predict(fit, X[te])
                coefs = fit["beta_std"]
            coef_rows += [{"model": m, "fold": k, "alpha": alpha, "predictor": pr, "coef_per_sd": float(c)}
                          for pr, c in zip(spec["predictors"], coefs)]
    oof = pd.DataFrame({GROUP: g, "event_id": df["event_id"].to_numpy(), "fold": fold, "observed": y})
    for m in pred:
        oof[f"pred_{m}"] = pred[m]
    return oof, pd.DataFrame(coef_rows)


def _verdict(d: dict) -> str:
    lo, hi = d["interval"]
    return "candidate lower error" if lo > 0 else "candidate higher error" if hi < 0 else "no detectable difference"


def _stability(coefs: pd.DataFrame, full: dict[str, float]) -> dict[str, dict[str, Any]]:
    out = {}
    for pr, c in coefs.groupby("predictor", sort=False)["coef_per_sd"]:
        ref = full[pr]
        out[pr] = {"full_data": ref, "fold_median": float(c.median()), "fold_min": float(c.min()),
                   "fold_max": float(c.max()), "share_same_sign_as_full": float((np.sign(c) == np.sign(ref)).mean()),
                   "sign_changes_across_folds": bool((c > 0).any() and (c < 0).any())}
    return out


def run_experiment(cfg: dict, rows_by_device: dict[str, pd.DataFrame]) -> RefinementResult:
    validate_config(cfg)
    names = list(cfg["models"])
    models_all = ["naive", *names]
    boot_cfg, level = cfg["bootstrap"], float(cfg["bootstrap"]["level"])
    metrics: dict[str, Any] = {"primary_metric": "participant_macro_mae", "level": level,
                               "improvement_definition": "MAE(reference) - MAE(candidate); positive = candidate lower error",
                               "devices": {}}
    meta: dict[str, Any] = {"models": cfg["models"], "ridge": cfg["ridge"], "devices": {}}
    oofs, splits, coefs_all = [], [], []
    for d_idx, d in enumerate(cfg["devices"]):
        df = rows_by_device[d]
        oof, coefs = lopo(df, cfg)
        agg = m4._participant_errors(oof, models_all)
        W = m4.bootstrap_weights(len(agg), int(boot_cfg["resamples"]), [int(boot_cfg["seed"]), d_idx])  # = M4 seeds
        bs = m4.bootstrap_statistics(agg, models_all, W)
        res: dict[str, Any] = {"n_breakfasts": len(df), "n_participants": int(df[GROUP].nunique()), "models": {},
                               "comparisons": []}
        for m in models_all:
            pm = m4.point_metrics(oof, m)
            pp = (oof["observed"] - oof[f"pred_{m}"]).abs().groupby(oof[GROUP]).mean()
            pm["participant_mae_p25"], pm["participant_mae_p75"] = float(pp.quantile(0.25)), float(pp.quantile(0.75))
            pm["intervals"] = {k: m4._interval(v, level) for k, v in bs[m].items()}
            res["models"][m] = pm
        for c in cfg["comparisons"]:
            cand, ref = c["candidate"], c["reference"]
            entry = {"candidate": cand, "reference": ref}
            for s in STAT_KEYS:
                dd = m4._difference(bs[ref][s], bs[cand][s], res["models"][ref][s] - res["models"][cand][s], level)
                dd.pop("improvement")
                dd["verdict"] = _verdict(dd)
                entry[s] = dd
            res["comparisons"].append(entry)
        # diagnostics: full-data VIF, alpha selection, coefficient stability
        diag: dict[str, Any] = {"vif": {}, "selected_alpha_counts": {}, "full_data_alpha": {}, "coefficients": {}}
        y, g = df[TARGET].to_numpy(float), df[GROUP].to_numpy()
        for m, spec in cfg["models"].items():
            X = df[spec["predictors"]].to_numpy(float)
            if spec["type"] == "ols":
                diag["vif"][m] = dict(zip(spec["predictors"], m4._vif(X)))
                full = dict(zip(spec["predictors"], _ols_std_coefficients(X, y)))
            else:
                a_full, _ = select_alpha(X, y, g, list(cfg["ridge"]["alphas"]))
                diag["full_data_alpha"][m] = a_full
                full = dict(zip(spec["predictors"], ridge_fit(X, y, a_full)["beta_std"]))
                sel = coefs[coefs.model == m].drop_duplicates("fold")["alpha"]
                diag["selected_alpha_counts"][m] = {repr(float(a)): int(n) for a, n in sel.value_counts().sort_index().items()}
            diag["coefficients"][m] = _stability(coefs[coefs.model == m], full)
        pw = df.drop_duplicates(GROUP)
        diag["participant_correlation_fasting_hba1c"] = {
            "n_participants": len(pw), "pearson": float(pw[ADDED].corr(pw["hba1c"])),
            "spearman": float(pw[ADDED].corr(pw["hba1c"], method="spearman"))}
        meta["devices"][d] = diag
        metrics["devices"][d] = res
        oofs.append(oof.assign(device=d))
        splits.append(oof.groupby([GROUP, "fold"], as_index=False).size()
                      .rename(columns={"size": "n_test_breakfasts"}).assign(device=d))
        coefs_all.append(coefs.assign(device=d))
    cols = ["device", GROUP, "event_id", "fold", "observed", *[f"pred_{m}" for m in models_all]]
    return RefinementResult(metrics, meta, pd.concat(oofs, ignore_index=True)[cols],
                            pd.concat(splits, ignore_index=True)[["device", GROUP, "fold", "n_test_breakfasts"]],
                            pd.concat(coefs_all, ignore_index=True)[["device", "model", "fold", "alpha", "predictor", "coef_per_sd"]])


def check_reference(res: RefinementResult, reference_oof: pd.DataFrame, devices: list[str]) -> dict[str, float]:
    """Model A and naive must reproduce the M4 primary out-of-fold predictions (same rows, folds and fits)."""
    out = {}
    for d in devices:
        ref = reference_oof[(reference_oof["analysis"] == "primary") & (reference_oof["device"] == d)]
        new = res.oof_predictions[res.oof_predictions["device"] == d]
        if (list(ref["event_id"]) != list(new["event_id"]) or list(ref["fold"]) != list(new["fold"])):
            raise ValueError(f"{d}: rows or folds differ from the M4 reference")
        diff = max(float(np.abs(ref["pred_linear"].to_numpy() - new[f"pred_{REFERENCE_MODEL}"].to_numpy()).max()),
                   float(np.abs(ref["pred_naive"].to_numpy() - new["pred_naive"].to_numpy()).max()))
        if diff > 1e-6:
            raise ValueError(f"{d}: model A / naive predictions differ from M4 by up to {diff}")
        out[d] = diff
    return out


# ------------------------------------------------------------------ report
DL = m4.DEVICE_LABEL
SHORT = {"A_ols_full": "A  OLS, 9 approved", "B_ols_reduced": "B  OLS, no energy",
         "C_ols_reduced_fasting": "C  OLS, no energy + fasting glucose", "D1_ridge_reduced": "D1 ridge, as B",
         "D2_ridge_reduced_fasting": "D2 ridge, as C", "naive": "Naive (training mean)"}


def render_report(cfg: dict, res: RefinementResult, prov: dict) -> str:
    M, meta, devs = res.metrics, res.model_metadata, cfg["devices"]
    names = list(cfg["models"])
    pct = f"{M['level']:.0%}"
    head = " | ".join(DL.get(d, d) for d in devs)
    sep = "|---" * (len(devs) + 1) + "|"
    n, ci = m4._n, m4._ci
    L = ["# Breakfast iAUC model refinement (M4): collinearity, fasting glucose, ridge", "",
         f"- Experiment `{cfg['experiment']}` v{cfg['version']}; config sha256 `{prov['config_sha256']}`; code `{prov['git_commit']}`",
         f"- Inputs: M3 analysis table sha256 `{prov['analysis_table_sha256']}`; M1 participant attributes sha256 "
         f"`{prov['attributes_sha256']}`; M4 reference predictions sha256 `{prov['reference_oof_sha256']}`.",
         "- Reference check: model A and the naive model reproduce the M4 primary out-of-fold predictions "
         "(maximum absolute difference " + "; ".join(f"{DL.get(d, d)} {v:.1e}" for d, v in prov["reference_diff"].items())
         + "). The M4 experiment and its outputs were read only.",
         f"- Validation: leave-one-participant-out per device; {pct} intervals from {cfg['bootstrap']['resamples']:,} "
         "bootstrap resamples of whole participants, the same resamples for every model and the same seeds as M4.", "",
         "## Scope and limitations", "",
         "- **Purpose:** to judge whether removing redundant energy information, adding fasting glucose or "
         "regularisation gives a more defensible model, not to maximise a score.",
         "- **Internal validation only.** Leave-one-participant-out estimates performance for held-out participants "
         "within CGMacros. It is not external validation and is not evidence of clinical generalisability.",
         "- **The target is dataset-derived** signed iAUC (0–120 min) from M3, computed from CGM values the authors "
         "document as linearly interpolated to one-minute rows; it is not a measured physiological response. All "
         "calculable breakfasts are kept, including negative values.",
         "- **Fasting glucose** (`fasting_glucose_lab`, mg/dL) is one laboratory value per participant, documented "
         "in the data dictionary as fasting and taken at study start. Fasting status is documented, not independently "
         "verifiable; the files record a collection time of day but no date, so its timing relative to individual "
         "breakfasts cannot be verified. It is **not** the per-breakfast baseline glucose (the mean CGM value in the "
         "15 min before each breakfast), which remains a separate predictor. It is strongly correlated with HbA1c "
         "(see diagnostics).",
         "- **Devices are analysed independently and never pooled**; they share participants and no device is primary.",
         "- Scaling and ridge selection are fitted only inside training folds (outer and inner); the held-out "
         "participant is never used for scaling, alpha selection or fitting.", "",
         "## Models", "", "| Model | Type | Predictors |", "|---|---|---|"]
    for m, spec in cfg["models"].items():
        L.append(f"| {SHORT.get(m, m)} | {spec['type']} | {', '.join(spec['predictors'])} |")
    L += ["", f"Ridge: alpha grid {', '.join(f'{a:g}' for a in cfg['ridge']['alphas'])}; standardised predictors; "
          "chosen by inner leave-one-participant-out on participant-macro MAE, ties to the larger alpha.", "",
          "## Primary metric: participant-macro MAE", "",
          f"mg/dL × min, point estimate [{pct} interval].", "", f"| Model | {head} |", sep]
    for m in ["naive", *names]:
        L.append(f"| {SHORT.get(m, m)} | " + " | ".join(
            ci(M["devices"][d]["models"][m]["participant_macro_mae"],
               M["devices"][d]["models"][m]["intervals"]["participant_macro_mae"]) for d in devs) + " |")
    L += ["", "## Paired comparisons", "",
          f"Improvement = MAE(reference) − MAE(candidate) in participant-macro MAE; positive means the candidate has "
          f"lower error. Point estimate [{pct} interval] and the pre-specified reading.", "",
          f"| Candidate vs reference | {head} |", sep]
    for i, c in enumerate(cfg["comparisons"]):
        L.append(f"| {SHORT.get(c['candidate'], c['candidate'])} vs {SHORT.get(c['reference'], c['reference']).split()[0]} | "
                 + " | ".join(
                     (lambda e: f"{ci(e['point'], e['interval'])} — {e['verdict']}")(
                         M["devices"][d]["comparisons"][i]["participant_macro_mae"]) for d in devs) + " |")
    L += ["", "## Secondary metrics", "",
          f"| Model | Device | Pooled MAE [{pct}] | RMSE [{pct}] | R² vs naive (skill) [{pct}] | Calibration slope | Calibration intercept |",
          "|---|---|---|---|---|---|---|"]
    for m in ["naive", *names]:
        for d in devs:
            x = M["devices"][d]["models"][m]
            r2 = "–" if x["r2_vs_naive"] is None else ci(x["r2_vs_naive"], x["intervals"]["r2_vs_naive"], 3)
            L.append(f"| {SHORT.get(m, m)} | {DL.get(d, d)} | {ci(x['pooled_mae'], x['intervals']['pooled_mae'])} | "
                     f"{ci(x['rmse'], x['intervals']['rmse'])} | {r2} | {n(x['calibration_slope'], 2)} | "
                     f"{n(x['calibration_intercept'])} |")
    L += ["", "## Error distribution across held-out participants", "",
          "Per-participant MAE (mg/dL × min): median [IQR] (min–max).", "", f"| Model | {head} |", sep]
    for m in ["naive", *names]:
        L.append(f"| {SHORT.get(m, m)} | " + " | ".join(
            (lambda x: f"{n(x['participant_mae_median'])} [{n(x['participant_mae_p25'])}–{n(x['participant_mae_p75'])}] "
                       f"({n(x['participant_mae_min'])}–{n(x['participant_mae_max'])})")(M["devices"][d]["models"][m])
            for d in devs) + " |")
    L += ["", "## Ridge regularisation strength", "",
          "Alpha selected in the 45 outer folds (alpha: number of folds), and on all rows (descriptive fit only).", "",
          f"| Model | {head} |", sep]
    for m in [x for x in names if cfg["models"][x]["type"] == "ridge"]:
        L.append(f"| {SHORT.get(m, m)} | " + " | ".join(
            ", ".join(f"{float(a):g}: {k}" for a, k in meta["devices"][d]["selected_alpha_counts"][m].items())
            + f" (all rows: {meta['devices'][d]['full_data_alpha'][m]:g})" for d in devs) + " |")
    L += ["", "## Collinearity", "", "Variance inflation factors on all rows of each device (OLS specifications).", ""]
    ols = [m for m in names if cfg["models"][m]["type"] == "ols"]
    preds = [p for p in ALLOWED_PREDICTORS if any(p in cfg["models"][m]["predictors"] for m in ols)]
    L += ["| Predictor | " + " | ".join(f"{m.split('_')[0]} {DL.get(d, d)}" for m in ols for d in devs) + " |",
          "|---" * (1 + len(ols) * len(devs)) + "|"]
    for p in preds:
        L.append(f"| {p} | " + " | ".join(n(meta["devices"][d]["vif"][m].get(p), 1) if p in meta["devices"][d]["vif"][m]
                                            else "–" for m in ols for d in devs) + " |")
    L += [""] + [f"- Participant-level correlation of fasting glucose with HbA1c ({DL.get(d, d)} participants, "
                 f"n = {c['n_participants']}): Pearson {c['pearson']:.2f}, Spearman {c['spearman']:.2f}."
                 for d in devs for c in [meta["devices"][d]["participant_correlation_fasting_hba1c"]]]
    L += ["", "## Coefficient stability across folds", "",
          "Coefficients per SD of each predictor (mg/dL × min per training-fold SD): all-rows fit; median and range "
          "over the 45 leave-one-participant-out fits; share of folds with the same sign as the all-rows fit. This shows "
          "sensitivity to leaving out one participant at a time; it is not a confidence interval. \"(sign varies)\" marks "
          "a sign that changes across folds."]
    for d in devs:
        co = meta["devices"][d]["coefficients"]
        flips = [p for p in co.get(REFERENCE_MODEL, {}) if p in co.get("B_ols_reduced", {})
                 and np.sign(co[REFERENCE_MODEL][p]["full_data"]) != np.sign(co["B_ols_reduced"][p]["full_data"])]
        L.append(f"- {DL.get(d, d)}: all-rows coefficient sign differs between A and B (removing energy) for: "
                 + (", ".join(flips) if flips else "none") + ".")
    for d in devs:
        L += ["", f"### {DL.get(d, d)}", "", "| Predictor | " + " | ".join(m.split("_")[0] for m in names) + " |",
              "|---" * (1 + len(names)) + "|"]
        for p in ALLOWED_PREDICTORS:
            cells = []
            for m in names:
                s = meta["devices"][d]["coefficients"][m].get(p)
                cells.append("–" if s is None else
                             f"{n(s['full_data'])} ({n(s['fold_min'])} to {n(s['fold_max'])}; {s['share_same_sign_as_full']:.0%})"
                             + (" (sign varies)" if s["sign_changes_across_folds"] else ""))
            L.append(f"| {p} | " + " | ".join(cells) + " |")
    L += ["", "## Prediction versus interpretation", "",
          "- Predictive performance is judged by out-of-fold error for held-out participants. Collinearity does not "
          "bias those predictions, but it makes individual coefficients unstable: correlated predictors (energy and "
          "the macronutrients; fasting glucose and HbA1c) can trade weight between them with little change in "
          "predictions.",
          "- Coefficients here are descriptive parameters of predictive models fitted on observational data. They are "
          "not independent or causal effects, and no clinical effect of any predictor is claimed — in particular "
          "not for coefficients whose sign changes across folds or that belong to collinear groups.",
          "- Ridge coefficients are shrunk by design and are not comparable in size with OLS coefficients.", "",
          "## Pre-specified reading", ""]
    q = {("B_ols_reduced", "A_ols_full"): "Removing total energy",
         ("C_ols_reduced_fasting", "B_ols_reduced"): "Adding fasting glucose to the reduced model",
         ("D1_ridge_reduced", "B_ols_reduced"): "Ridge on the reduced model",
         ("D2_ridge_reduced_fasting", "C_ols_reduced_fasting"): "Ridge on the fasting-glucose model"}
    for i, c in enumerate(cfg["comparisons"]):
        key = (c["candidate"], c["reference"])
        if key in q:
            L.append(f"- **{q[key]}** ({key[0].split('_')[0]} vs {key[1].split('_')[0]}): " + "; ".join(
                f"{DL.get(d, d)} {M['devices'][d]['comparisons'][i]['participant_macro_mae']['verdict']}" for d in devs) + ".")
    L += ["", "'No detectable difference' is not evidence of equivalence: no non-inferiority margin was set.",
          "", "Participant-level files (`split_assignments`, `oof_predictions`, `fold_coefficients`) must stay local.", ""]
    return "\n".join(L)


# ------------------------------------------------------------------ run
def run(config_path: str | Path, task_dir: str | Path, m1_dir: str | Path, reference_dir: str | Path,
        out_dir: str | Path, base_dir: str | Path) -> RefinementResult:
    cfg = load_config(config_path)
    reference_cfg = m4.load_config(Path(base_dir) / cfg["reference"]["config"])
    validate_config(cfg, reference_cfg)
    task, m1, ref_dir, out = Path(task_dir), Path(m1_dir), Path(reference_dir), Path(out_dir)
    if out.resolve() == ref_dir.resolve():
        raise ValueError("the refinement must not write into the M4 reference output directory")
    paths = {"analysis_table": task / "analysis_table.parquet", "task_manifest": task / "task_manifest.json",
             "attributes": m1 / "participant_attributes.parquet", "m1_manifest": m1 / "_provenance" / "manifest.json",
             "reference_oof": ref_dir / "oof_predictions.parquet",
             "reference_manifest": ref_dir / "experiment_manifest.json"}
    for p in paths.values():
        if not p.exists():
            raise FileNotFoundError(f"input not found: {p}")
    sha = {k: sha256_file(p) for k, p in paths.items() if p.suffix == ".parquet"}

    def recorded(manifest_path: Path, suffix: str) -> list[str]:
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        return [o["sha256"] for o in man["outputs"] if o["path"].endswith(suffix)]
    for key, man, suffix in (("analysis_table", "task_manifest", "analysis_table.parquet"),
                             ("attributes", "m1_manifest", "participant_attributes.parquet"),
                             ("reference_oof", "reference_manifest", "oof_predictions.parquet")):
        if recorded(paths[man], suffix) != [sha[key]]:
            raise ValueError(f"{suffix} hash does not match {paths[man].name} (lineage broken)")
    ref_man = json.loads(paths["reference_manifest"].read_text(encoding="utf-8"))
    if ref_man["lineage"]["analysis_table_sha256"] != sha["analysis_table"]:
        raise ValueError("the M4 reference was run on a different analysis table")
    m1_sha = sha256_file(paths["m1_manifest"])
    if ref_man["lineage"]["m1_manifest_sha256"] != m1_sha:
        raise ValueError("the M4 reference was run on a different M1 run")

    table = pd.read_parquet(paths["analysis_table"])
    rows = m4.select_rows(table, cfg["devices"], "all_calculable")          # = M4 primary rows
    attrs = pd.read_parquet(paths["attributes"])
    rows = {d: add_attributes(df, attrs, cfg) for d, df in rows.items()}
    res = run_experiment(cfg, rows)
    ref_diff = check_reference(res, pd.read_parquet(paths["reference_oof"]), cfg["devices"])

    out.mkdir(parents=True, exist_ok=True)
    manifest = RunManifest(f"experiment:{cfg['experiment']}", base_dir)
    for p in [*paths.values(), Path(config_path), Path(base_dir) / cfg["reference"]["config"]]:
        manifest.add_input(p)
    manifest.add_config("experiment", cfg)
    manifest.set_source({"experiment": cfg["experiment"], "version": cfg["version"],
                         "reference_experiment": reference_cfg["experiment"]})
    manifest.data["environment"] = m4._versions()
    manifest.data["lineage"] = {
        "m1_manifest_sha256": m1_sha, "analysis_table_sha256": sha["analysis_table"],
        "participant_attributes_sha256": sha["attributes"], "reference_oof_sha256": sha["reference_oof"],
        "reference_check_max_abs_diff": ref_diff,
        "row_selection": "M4 primary rows: outcome_status == 'calculable', one device at a time",
        "added_columns": {ADDED: "M1 participant_attributes, unit mg/dL exactly, one value per participant"},
        "column_selection": {m: s["predictors"] for m, s in cfg["models"].items()},
        "transformations": "ridge only: standardisation with training-fold mean/SD (outer and inner folds); "
                           "no imputation, encoding or outcome transformation"}
    for name, df in (("split_assignments", res.split_assignments), ("oof_predictions", res.oof_predictions),
                     ("fold_coefficients", res.fold_coefficients)):
        path = out / f"{name}.parquet"
        df.to_parquet(path, index=False)
        manifest.add_output(path, table=name, rows=len(df))
    meta = dict(res.model_metadata, environment=m4._versions(), bootstrap=cfg["bootstrap"],
                validation=cfg["validation"], comparisons=cfg["comparisons"])
    for name, obj in (("metrics.json", res.metrics), ("model_metadata.json", meta)):
        (out / name).write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
        manifest.add_output(out / name)
    report = render_report(cfg, res, {"config_sha256": config_hash(cfg), "git_commit": manifest.data["git_commit"],
                                      "analysis_table_sha256": sha["analysis_table"],
                                      "attributes_sha256": sha["attributes"],
                                      "reference_oof_sha256": sha["reference_oof"], "reference_diff": ref_diff})
    (out / "refinement_report.md").write_text(report, encoding="utf-8")
    manifest.add_output(out / "refinement_report.md")
    manifest.write(out / "refinement_manifest.json")
    return res


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m pghd_ready.conditions.cardiometabolic.experiments.breakfast_iauc_refinement",
                                 description="M4 refinement: collinearity, fasting glucose and ridge (internal LOPO validation).")
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--task-dir", required=True, type=Path, help="M3 task output directory")
    ap.add_argument("--m1", required=True, type=Path, help="M1 output directory (participant attributes)")
    ap.add_argument("--reference-dir", required=True, type=Path, help="M4 reference experiment outputs (read only)")
    ap.add_argument("--out", required=True, type=Path, help="output directory (git-ignored, not the reference)")
    ap.add_argument("--base-dir", type=Path, default=Path.cwd())
    args = ap.parse_args(argv)
    try:
        res = run(args.config, args.task_dir, args.m1, args.reference_dir, args.out, args.base_dir)
    except Exception as exc:
        print(f"Refinement FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Refinement SUCCESS: {len(res.oof_predictions)} out-of-fold predictions")
    print(f"Report: {args.out / 'refinement_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
