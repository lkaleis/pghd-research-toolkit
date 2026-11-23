"""Independent reference profiler for CGMacros (used only by real-data reconciliation tests).

Deliberately independent of the ingestion engine:
  * reads files with ``pandas.read_csv`` (the engine uses the ``csv`` module),
  * takes documented ranges from the official data dictionaries (the engine uses our YAML),
  * computes simple aggregates per participant and column.

It returns aggregates only. No participant-level data is written anywhere.
"""

from __future__ import annotations

import re
from collections import Counter
from decimal import Decimal
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
CGM_ROOT = REPO_ROOT / "data" / "raw" / "cgmacros"
DOCS_ROOT = REPO_ROOT / "data" / "raw" / "cgmacros_docs"

# source column (as in the files, stripped) -> engine stream id
TS_STREAMS = {"Libre GL": "libre_pro", "Dexcom GL": "dexcom_g6", "HR": "fitbit_hr",
              "Calories (Activity)": "fitbit_activity_kcal", "METs": "fitbit_met", "Intensity": "fitbit_intensity",
              "Steps": "fitbit_steps"}
MEAL_ATTRS = {"Calories": "energy_kcal", "Carbs": "carbohydrate_g", "Protein": "protein_g", "Fat": "fat_g",
              "Fiber": "fiber_g", "Amount Consumed": "amount_consumed_raw"}
BIO_CONCEPTS = {"Age": "age", "BMI": "body_mass_index", "Body weight": "body_weight", "Height": "body_height",
                "A1c PDL (Lab)": "hba1c", "Fasting GLU - PDL (Lab)": "fasting_glucose_lab", "Insulin": "fasting_insulin",
                "Triglycerides": "triglycerides", "Cholesterol": "total_cholesterol", "HDL": "hdl_cholesterol",
                "Non HDL": "non_hdl_cholesterol", "LDL (Cal)": "ldl_cholesterol_calculated",
                "VLDL (Cal)": "vldl_cholesterol_calculated", "Cho/HDL Ratio": "cholesterol_hdl_ratio"}


def available() -> bool:
    return CGM_ROOT.is_dir() and (DOCS_ROOT / "DataDictionary_CGMacros-00X.csv").is_file()


def _half_unit(text: str) -> Decimal:
    """Half a unit in the last written decimal place, derived from the dictionary text itself."""
    decimals = len(text.split(".")[1]) if "." in text else 0
    return Decimal(1).scaleb(-decimals) / 2


def documented_ranges(dictionary_csv: Path) -> dict[str, tuple[float, float]]:
    """Parse 'Numeric: a-b' ranges from an official data dictionary (field names lower-cased, stripped).

    Bounds are widened by half a unit in the last decimal place written in the dictionary (precision tolerance).
    """
    dd = pd.read_csv(dictionary_csv, encoding="utf-8-sig")
    out = {}
    for field, rng in zip(dd["Field"], dd["Range"]):
        m = re.search(r"numeric:\s*([\d.]+)\s*-\s*([\d.]+)", str(rng), re.IGNORECASE)  # dictionary mixes case
        if m:
            lo, hi = m.group(1), m.group(2)
            out[str(field).strip().lower()] = (float(Decimal(lo) - _half_unit(lo)), float(Decimal(hi) + _half_unit(hi)))
    return out


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s[s.str.strip() != ""], errors="coerce")


def profile() -> dict:
    ts_ranges = documented_ranges(DOCS_ROOT / "DataDictionary_CGMacros-00X.csv")  # keys: 'libre gl', 'mets', ...
    files, streams, meals, ts_bounds = {}, {}, {}, {}
    for path in sorted(CGM_ROOT.glob("CGMacros-*/CGMacros-*.csv")):
        pid = "cgmacros:" + path.parent.name.split("-")[1]
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        df.columns = [c.strip() for c in df.columns]
        files[path.relative_to(CGM_ROOT).as_posix()] = {"rows": len(df),
                                                        "empty_timestamps": int((df["Timestamp"] == "").sum())}
        t = pd.to_datetime(df["Timestamp"], format="%Y-%m-%d %H:%M:%S")
        ts_bounds[pid] = (t.min(), t.max())
        for col, stream in TS_STREAMS.items():
            if col not in df.columns:
                continue
            v = _num(df[col])
            rng = ts_ranges.get(col.lower())
            streams[(pid, stream)] = {
                "n": int(v.notna().sum()), "sum": float(v.sum()),
                "at_limit": int(v.isin([40.0, 400.0]).sum()) if stream in ("libre_pro", "dexcom_g6") else None,
                "outside": int(((v < rng[0]) | (v > rng[1])).sum()) if rng else None,
            }
        meal_rows = df[df["Meal Type"].str.strip() != ""]
        m = {"n": len(meal_rows), "labels": Counter(meal_rows["Meal Type"]),
             "photos": int((df["Image path"].str.strip() != "").sum())}
        for col, target in MEAL_ATTRS.items():
            if col in df.columns:
                v = _num(meal_rows[col])
                rng = ts_ranges.get(col.lower())
                m[target] = {"n": int(v.notna().sum()), "sum": float(v.sum()),
                             "below": int((v < rng[0]).sum()) if rng else None,
                             "above": int((v > rng[1]).sum()) if rng else None,
                             "outside": int(((v < rng[0]) | (v > rng[1])).sum()) if rng else None}
        meals[pid] = m

    bio_ranges = documented_ranges(DOCS_ROOT / "DataDictionary_Bio.csv")
    bio = pd.read_csv(CGM_ROOT / "bio.csv", dtype=str, keep_default_na=False, encoding="utf-8-sig")
    bio.columns = [c.strip() for c in bio.columns]  # duplicate 'Time (t)' become 'Time (t).1', '.2' (pandas)
    attrs = {}
    for col, concept in BIO_CONCEPTS.items():
        v = _num(bio[col])
        rng = bio_ranges.get(col.lower())
        attrs[concept] = {"n": int(v.notna().sum()), "sum": float(v.sum()),
                          "outside": int(((v < rng[0]) | (v > rng[1])).sum()) if rng else None,
                          "values": v.dropna().tolist()}
    fs_cols = [c for c in bio.columns if "Fingerstick" in c]
    fs = pd.concat([_num(bio[c]) for c in fs_cols])
    attrs["capillary_glucose"] = {"n": int(fs.notna().sum()), "sum": float(fs.sum()), "outside": None}
    for col, concept in (("Gender", "sex"), ("Self-identify", "self_identified_race_ethnicity")):
        attrs[concept] = {"n": int((bio[col].str.strip() != "").sum()), "labels": Counter(bio[col].str.strip())}
    return {"files": files, "streams": streams, "meals": meals, "ts_bounds": ts_bounds, "attributes": attrs,
            "bio_participants": int((bio["subject"].str.strip() != "").sum())}
