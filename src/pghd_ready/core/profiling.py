"""Descriptive data-quality profiling of M1 time-series observations (M2b, first milestone).

Profiles are **descriptive only**: they count and summarise what is in ``observations.parquet``. They
never modify, interpolate, resample or impute observations, produce no findings, and set no thresholds.

Metric definitions (all computed per participant x stream unless stated otherwise)
---------------------------------------------------------------------------------
* ``n_observations``: rows in ``observations`` for the participant and stream.
* ``n_value_missing``: rows whose standardised ``value`` is empty (e.g. nulled by an M1 documented issue).
* ``n_timestamp_missing``: rows without a timestamp; they are excluded from all temporal metrics.
* ``n_distinct_timestamps`` / ``n_duplicate_timestamp_rows``: distinct timestamps, and rows beyond the first
  at an already-seen timestamp. Interval and gap metrics use distinct timestamps.
* ``source_rows`` / ``value_present_fraction``: rows in the participant's source file(s) (M1 file stats) and
  the fraction of them that produced an observation for this stream. Meaningful for wide files in which
  each row is one time point; it describes the file, not the device.
* ``first_timestamp`` / ``last_timestamp`` / ``observed_span_hours``: earliest and latest observed timestamp
  and the time between them. **This is observed timestamp coverage, not device wear-time**: readings do not
  show that a device was worn, and their absence does not show that it was not.
* ``calendar_days_spanned`` / ``days_with_observations``: calendar dates from first to last observation
  (inclusive), and dates with at least one observation. Dates are as recorded (they may be shifted).
* Intervals: differences between consecutive distinct timestamps (seconds): min, 5th percentile, median,
  95th percentile, max, the most frequent (modal) interval and its share. Observed only; a regular
  pattern is never taken as evidence of sampling origin.
* ``expected_interval_s`` / ``expected_interval_basis``: the expected sampling interval, only if it is
  documented in the source configuration (``documented``, M1 ``declared_native_interval_s``) or supplied
  explicitly by the analyst (``configured``). Otherwise ``none`` and no gap or slot metric is computed.
  Intervals are never inferred from the data.
* Gaps (only with an expected interval E and the gap factor F, default 1.5): a gap is an interval between
  consecutive distinct timestamps strictly longer than F x E. With F = 1.5 a reading arriving within half
  an interval of its expected time counts as on time; beyond that at least one expected reading is absent.
  ``n_gaps``, ``total_gap_hours`` (sum of gap intervals), ``longest_gap_hours``.
* Slot coverage (only with an expected interval E): the observed span is divided into consecutive slots of
  length E starting at the first timestamp; ``slot_coverage`` is the fraction of slots containing at least
  one observation. Unlike a count ratio it is not inflated by readings denser than E (for example values
  interpolated onto a finer grid). Daily: ``slot_coverage_of_day`` uses slots of length E from midnight.
* Sampling origin: per stream and participant, readings by origin (native / interpolated / unknown) and,
  for native and interpolated, by how the origin was established (declared in configuration, inferred by
  a named classification transform, or unattributed), via :mod:`pghd_ready.core.sampling`.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from . import sampling as _sampling
from .provenance import RunManifest

DEFAULT_GAP_FACTOR = 1.5
KEY = ["participant_id", "stream_id"]
ORIGIN_COLUMNS = ["native_declared", "native_inferred", "native_unattributed", "interpolated_declared",
                  "interpolated_inferred", "interpolated_unattributed", "unknown"]


@dataclass
class Profile:
    stream_profile: pd.DataFrame          # participant x stream (participant-level; keep local)
    daily_profile: pd.DataFrame           # participant x stream x calendar date (participant-level)
    gaps: pd.DataFrame                    # one row per gap (participant-level)
    dataset_summary: pd.DataFrame         # one row per stream (aggregate)
    parameters: dict[str, Any] = field(default_factory=dict)


# ------------------------------------------------------------------ expected intervals
def expected_intervals(streams: pd.DataFrame, configured: Mapping[str, float] | None = None) -> pd.DataFrame:
    """Expected interval per stream: documented (M1 declaration) or configured by the analyst; never inferred.

    A configured interval that contradicts a documented one is an error, not an override.
    """
    configured = dict(configured or {})
    rows = []
    stream_ids = sorted(set(streams["stream_id"]) | set(configured))
    for sid in stream_ids:
        declared = streams.loc[streams["stream_id"] == sid, "declared_native_interval_s"].dropna().unique()
        if len(declared) > 1:
            raise ValueError(f"Stream {sid!r} has conflicting documented intervals {sorted(declared)}.")
        doc = float(declared[0]) if len(declared) else None
        conf = configured.get(sid)
        if conf is not None:
            conf = float(conf)
            if not (math.isfinite(conf) and conf > 0):
                raise ValueError(f"Configured interval for {sid!r} must be a positive number of seconds.")
            if doc is not None and conf != doc:
                raise ValueError(f"Configured interval {conf:g} s for {sid!r} contradicts the documented {doc:g} s.")
        if doc is not None:
            rows.append({"stream_id": sid, "expected_interval_s": doc, "expected_interval_basis": "documented"})
        elif conf is not None:
            rows.append({"stream_id": sid, "expected_interval_s": conf, "expected_interval_basis": "configured"})
        else:
            rows.append({"stream_id": sid, "expected_interval_s": np.nan, "expected_interval_basis": "none"})
    return pd.DataFrame(rows, columns=["stream_id", "expected_interval_s", "expected_interval_basis"])


# ------------------------------------------------------------------ profiling
def _origin_table(obs: pd.DataFrame, log: list[dict] | None) -> pd.DataFrame:
    labels = _sampling.origin_labels(obs["is_interpolated"])
    evidence = _sampling.origin_evidence(obs, log or [])
    key = np.where(labels == _sampling.UNKNOWN, "unknown", labels + "_" + evidence)
    t = pd.DataFrame({"participant_id": obs["participant_id"].to_numpy(), "stream_id": obs["stream_id"].to_numpy(),
                      "k": key})
    out = t.groupby(KEY + ["k"]).size().unstack("k", fill_value=0)
    return out.reindex(columns=ORIGIN_COLUMNS, fill_value=0).add_prefix("origin_").reset_index()


def profile_observations(observations: pd.DataFrame, streams: pd.DataFrame,
                         file_stats: pd.DataFrame | None = None, log: list[dict] | None = None,
                         configured_intervals: Mapping[str, float] | None = None,
                         gap_factor: float = DEFAULT_GAP_FACTOR) -> Profile:
    """Profile M1 observations (read-only). See the module docstring for every metric's definition."""
    if not (math.isfinite(gap_factor) and gap_factor >= 1):
        raise ValueError("gap_factor must be a finite number >= 1.")
    exp = expected_intervals(streams, configured_intervals)
    obs = observations
    base = obs.groupby(KEY, sort=True).agg(
        n_observations=("stream_id", "size"),
        n_value_missing=("value", lambda v: int(v.isna().sum())),
        n_timestamp_missing=("timestamp", lambda v: int(v.isna().sum())))

    ts = obs.loc[obs["timestamp"].notna(), KEY + ["timestamp"]]
    distinct = ts.drop_duplicates().sort_values(KEY + ["timestamp"], kind="mergesort").reset_index(drop=True)
    dup = (ts.groupby(KEY).size() - distinct.groupby(KEY).size()).rename("n_duplicate_timestamp_rows")
    temporal = distinct.groupby(KEY).agg(n_distinct_timestamps=("timestamp", "size"),
                                         first_timestamp=("timestamp", "min"), last_timestamp=("timestamp", "max"))
    temporal["observed_span_hours"] = (temporal["last_timestamp"] - temporal["first_timestamp"]).dt.total_seconds() / 3600
    temporal["calendar_days_spanned"] = ((temporal["last_timestamp"].dt.normalize()
                                          - temporal["first_timestamp"].dt.normalize()).dt.days + 1)
    temporal["days_with_observations"] = distinct.assign(d=distinct["timestamp"].dt.normalize()).groupby(KEY)["d"].nunique()

    # intervals between consecutive distinct timestamps
    same = (distinct["participant_id"].eq(distinct["participant_id"].shift())
            & distinct["stream_id"].eq(distinct["stream_id"].shift()))
    iv = distinct.assign(interval_s=distinct["timestamp"].diff().dt.total_seconds(),
                         prev=distinct["timestamp"].shift())[same.to_numpy()]
    g = iv.groupby(KEY)["interval_s"]
    ivs = pd.DataFrame({"n_intervals": g.size(), "interval_min_s": g.min(), "interval_p05_s": g.quantile(0.05),
                        "interval_median_s": g.median(), "interval_p95_s": g.quantile(0.95),
                        "interval_max_s": g.max()})
    ivs["interval_mode_s"] = np.nan
    ivs["interval_mode_share"] = np.nan
    if len(iv):
        vc = iv.groupby(KEY + ["interval_s"]).size().rename("n").reset_index()
        vc = vc.sort_values(KEY + ["n", "interval_s"], ascending=[True, True, False, True], kind="mergesort")
        mode = vc.groupby(KEY).head(1).set_index(KEY)
        ivs["interval_mode_s"] = mode["interval_s"]
        ivs["interval_mode_share"] = mode["n"] / ivs["n_intervals"]

    prof = base.join(dup).join(temporal).join(ivs).reset_index()
    prof["n_duplicate_timestamp_rows"] = prof["n_duplicate_timestamp_rows"].fillna(0).astype(int)
    prof = prof.merge(exp, on="stream_id", how="left")
    prof["expected_interval_basis"] = prof["expected_interval_basis"].fillna("none")

    # completeness against the source file rows (wide files: one row per time point)
    prof["source_rows"] = np.nan
    if file_stats is not None and len(file_stats) and "source_file" in obs:
        files = obs[["participant_id", "source_file"]].drop_duplicates()
        n_rec = file_stats.groupby("source_file")["n_records"].sum()
        per_pid = files.assign(n=files["source_file"].map(n_rec)).groupby("participant_id")["n"].sum(min_count=1)
        prof["source_rows"] = prof["participant_id"].map(per_pid)
    prof["value_present_fraction"] = (prof["n_observations"] - prof["n_value_missing"]) / prof["source_rows"]

    # gaps and slot coverage, only where an expected interval exists
    e_map = exp.set_index("stream_id")["expected_interval_s"]
    iv = iv.assign(expected=iv["stream_id"].map(e_map))
    is_gap = iv["expected"].notna() & (iv["interval_s"] > gap_factor * iv["expected"])
    gaps = iv.loc[is_gap, KEY + ["prev", "timestamp", "interval_s", "expected"]].rename(columns={
        "prev": "gap_start", "timestamp": "gap_end", "interval_s": "gap_s", "expected": "expected_interval_s"})
    gaps = gaps.reset_index(drop=True)
    gsum = gaps.groupby(KEY).agg(n_gaps=("gap_s", "size"), total_gap_s=("gap_s", "sum"), longest_gap_s=("gap_s", "max"))
    prof = prof.merge(gsum.reset_index(), on=KEY, how="left")
    has_e = prof["expected_interval_s"].notna()
    prof.loc[has_e, "n_gaps"] = prof.loc[has_e, "n_gaps"].fillna(0)
    prof.loc[has_e, ["total_gap_s", "longest_gap_s"]] = prof.loc[has_e, ["total_gap_s", "longest_gap_s"]].fillna(0)
    prof["total_gap_hours"] = prof.pop("total_gap_s") / 3600
    prof["longest_gap_hours"] = prof.pop("longest_gap_s") / 3600
    prof["n_gaps"] = prof["n_gaps"].astype("Int64")

    d = distinct.assign(expected=distinct["stream_id"].map(e_map))
    d = d[d["expected"].notna()]
    first = d.groupby(KEY)["timestamp"].transform("min")
    d = d.assign(slot=np.floor((d["timestamp"] - first).dt.total_seconds() / d["expected"]))
    slots = d.groupby(KEY).agg(n_slots_observed=("slot", "nunique")).reset_index()
    prof = prof.merge(slots, on=KEY, how="left")
    prof["n_slots"] = np.floor(prof["observed_span_hours"] * 3600 / prof["expected_interval_s"]) + 1
    prof["slot_coverage"] = prof["n_slots_observed"] / prof["n_slots"]
    for c in ("n_slots", "n_slots_observed"):
        prof[c] = prof[c].astype("Int64")

    prof = prof.merge(_origin_table(obs, log), on=KEY, how="left")

    # daily profile
    dd = distinct.assign(date=distinct["timestamp"].dt.normalize(), expected=distinct["stream_id"].map(e_map))
    rows_per_day = ts.assign(date=ts["timestamp"].dt.normalize()).groupby(KEY + ["date"]).size().rename("n_observations")
    daily = dd.groupby(KEY + ["date"]).agg(n_distinct_timestamps=("timestamp", "size"),
                                            first_timestamp=("timestamp", "min"), last_timestamp=("timestamp", "max"))
    daily = rows_per_day.to_frame().join(daily)
    daily["observed_span_hours"] = (daily["last_timestamp"] - daily["first_timestamp"]).dt.total_seconds() / 3600
    sod = (dd["timestamp"] - dd["date"]).dt.total_seconds()
    dd = dd.assign(slot=np.floor(sod / dd["expected"]))
    dslots = dd[dd["expected"].notna()].groupby(KEY + ["date"])["slot"].nunique().rename("n_slots_observed")
    daily = daily.join(dslots).reset_index().merge(exp, on="stream_id", how="left")
    daily["slots_per_day"] = np.ceil(86400 / daily["expected_interval_s"])
    daily["slot_coverage_of_day"] = daily["n_slots_observed"] / daily["slots_per_day"]
    for c in ("n_slots_observed", "slots_per_day"):
        daily[c] = daily[c].astype("Int64")

    params = {"gap_factor": gap_factor,
              "expected_intervals": {r.stream_id: {"seconds": None if pd.isna(r.expected_interval_s)
                                                   else r.expected_interval_s, "basis": r.expected_interval_basis}
                                     for r in exp.itertuples()}}
    return Profile(stream_profile=prof, daily_profile=daily, gaps=gaps,
                   dataset_summary=summarise(prof), parameters=params)


def summarise(prof: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across participants, one row per stream (no participant identifiers)."""
    def q(s: pd.Series, p: float) -> float:
        s = s.dropna()
        return float(s.quantile(p)) if len(s) else np.nan

    rows = []
    for sid, g in prof.groupby("stream_id", sort=True):
        r: dict[str, Any] = {
            "stream_id": sid, "n_participants": int(g["participant_id"].nunique()),
            "n_observations": int(g["n_observations"].sum()), "n_value_missing": int(g["n_value_missing"].sum()),
            "n_timestamp_missing": int(g["n_timestamp_missing"].sum()),
            "n_duplicate_timestamp_rows": int(g["n_duplicate_timestamp_rows"].sum()),
            "expected_interval_s": g["expected_interval_s"].iloc[0],
            "expected_interval_basis": g["expected_interval_basis"].iloc[0],
            "observed_span_days_median": q(g["observed_span_hours"], 0.5) / 24,
            "observed_span_days_min": g["observed_span_hours"].min() / 24,
            "observed_span_days_max": g["observed_span_hours"].max() / 24,
            "days_with_observations_median": q(g["days_with_observations"], 0.5),
            "value_present_fraction_median": q(g["value_present_fraction"], 0.5),
            "interval_median_s_median": q(g["interval_median_s"], 0.5),
            "interval_mode_s_most_common": (g["interval_mode_s"].mode().min() if g["interval_mode_s"].notna().any()
                                            else np.nan),
            "interval_max_s_max": g["interval_max_s"].max(),
        }
        if g["expected_interval_s"].notna().any():
            r.update({
                "n_gaps_total": int(g["n_gaps"].fillna(0).sum()),
                "participants_with_gaps": int((g["n_gaps"].fillna(0) > 0).sum()),
                "longest_gap_hours_median": q(g["longest_gap_hours"], 0.5),
                "longest_gap_hours_max": g["longest_gap_hours"].max(),
                "slot_coverage_median": q(g["slot_coverage"], 0.5),
                "slot_coverage_min": g["slot_coverage"].min(), "slot_coverage_max": g["slot_coverage"].max(),
            })
        for c in ORIGIN_COLUMNS:
            r[f"origin_{c}"] = int(g[f"origin_{c}"].fillna(0).sum())
        rows.append(r)
    out = pd.DataFrame(rows)
    for c in ("n_gaps_total", "participants_with_gaps"):  # absent (NA) where no expected interval exists
        if c in out:
            out[c] = out[c].astype("Int64")
    return out


# ------------------------------------------------------------------ run on M1 outputs
def read_transform_log(path: str | Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _fmt(v: Any, digits: int = 2) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)) or v is pd.NA:
        return "n/a"
    if isinstance(v, (float, np.floating)):
        return f"{v:,.{digits}f}"
    if isinstance(v, (int, np.integer)):
        return f"{v:,}"
    return str(v)


def render_report(profile: Profile, provenance: Mapping[str, Any]) -> str:
    """Analyst-readable, aggregate-only report (no participant identifiers)."""
    s = profile.dataset_summary
    lines = ["# PGHD-Ready data-quality profile (M2b, descriptive)", "",
             "Descriptive counts of what was ingested. Nothing here is a finding, a threshold or a judgement of "
             "fitness for any analysis. Observations were not modified, interpolated, resampled or imputed.", "",
             f"- M1 input: `{provenance.get('m1_dir')}` (M1 manifest sha256 `{provenance.get('m1_manifest_sha256')}`)",
             f"- Code: `{provenance.get('git_commit')}`",
             f"- Gap factor: {profile.parameters['gap_factor']:g} (a gap is an interval longer than factor x expected interval)",
             "", "## Expected sampling intervals", "",
             "| Stream | Expected interval | Basis |", "|---|---|---|"]
    for sid, e in profile.parameters["expected_intervals"].items():
        sec = e["seconds"]
        lines.append(f"| {sid} | {'not available' if sec is None else f'{sec:g} s'} | {e['basis']} |")
    lines += ["", "Gaps and slot coverage are computed only for streams with a documented or explicitly configured "
              "interval. Intervals are never inferred from the data.", "",
              "## Completeness and temporal coverage (across participants)", "",
              "| Stream | Participants | Observations | Value missing | Duplicate-timestamp rows | Value present in source rows (median) | Observed span, days (median [min-max]) | Days with observations (median) |",
              "|---|---|---|---|---|---|---|---|"]
    for r in s.itertuples():
        lines.append(f"| {r.stream_id} | {_fmt(r.n_participants)} | {_fmt(r.n_observations)} | {_fmt(r.n_value_missing)} | "
                     f"{_fmt(r.n_duplicate_timestamp_rows)} | {_fmt(r.value_present_fraction_median, 3)} | "
                     f"{_fmt(r.observed_span_days_median)} [{_fmt(r.observed_span_days_min)}-{_fmt(r.observed_span_days_max)}] | "
                     f"{_fmt(r.days_with_observations_median, 1)} |")
    lines += ["", "Observed span is the time between the first and last observed timestamp. It is **not device "
              "wear-time**: readings do not establish that a device was worn, and missing readings do not establish "
              "that it was not.", "", "## Sampling regularity and gaps", "",
              "| Stream | Median interval, s (median across participants) | Most common modal interval, s | Longest interval, s | Gaps (total) | Participants with gaps | Longest gap, h (median / max) | Slot coverage (median [min-max]) |",
              "|---|---|---|---|---|---|---|---|"]
    for r in s.itertuples():
        has = not pd.isna(getattr(r, "slot_coverage_median", np.nan))
        lines.append(f"| {r.stream_id} | {_fmt(r.interval_median_s_median, 0)} | {_fmt(r.interval_mode_s_most_common, 0)} | "
                     f"{_fmt(r.interval_max_s_max, 0)} | "
                     + (f"{_fmt(r.n_gaps_total)} | {_fmt(r.participants_with_gaps)} | "
                        f"{_fmt(r.longest_gap_hours_median)} / {_fmt(r.longest_gap_hours_max)} | "
                        f"{_fmt(r.slot_coverage_median, 3)} [{_fmt(r.slot_coverage_min, 3)}-{_fmt(r.slot_coverage_max, 3)}] |"
                        if has else "not assessed | not assessed | not assessed | not assessed |"))
    lines += ["", "Slot coverage: the observed span is split into consecutive slots of the expected interval; coverage "
              "is the fraction of slots with at least one observation. It is not inflated by readings denser than "
              "the expected interval.", "", "## Sampling origin", "",
              "| Stream | Native (declared) | Native (inferred) | Interpolated (declared) | Interpolated (inferred) | Unattributed | Unknown |",
              "|---|---|---|---|---|---|---|"]
    for r in s.itertuples():
        unattr = r.origin_native_unattributed + r.origin_interpolated_unattributed
        lines.append(f"| {r.stream_id} | {_fmt(r.origin_native_declared)} | {_fmt(r.origin_native_inferred)} | "
                     f"{_fmt(r.origin_interpolated_declared)} | {_fmt(r.origin_interpolated_inferred)} | "
                     f"{_fmt(unattr)} | {_fmt(r.origin_unknown)} |")
    lines += ["", "Origins are reported as recorded by M1. A regular timestamp pattern is never used to infer origin.",
              "", "## Definitions and limitations", "",
              "Full metric definitions are in `pghd_ready.core.profiling`. Participant-level tables "
              "(`stream_profile`, `daily_profile`, `gaps`) are written next to this report and must stay local.",
              "Agreement between streams measuring the same concept is not profiled in this milestone.", ""]
    return "\n".join(lines)


def profile_m1_run(m1_dir: str | Path, out_dir: str | Path, base_dir: str | Path,
                   configured_intervals: Mapping[str, float] | None = None,
                   gap_factor: float = DEFAULT_GAP_FACTOR) -> Profile:
    """Profile a finished M1 run and write tables, a manifest and an aggregate report to ``out_dir``."""
    m1, out = Path(m1_dir), Path(out_dir)
    prov = m1 / "_provenance"
    inputs = [m1 / "observations.parquet", m1 / "streams.parquet", prov / "file_stats.csv",
              prov / "transform_log.jsonl", prov / "manifest.json"]
    for p in inputs:
        if not p.exists():
            raise FileNotFoundError(f"M1 output not found: {p}")
    obs = pd.read_parquet(inputs[0])
    streams = pd.read_parquet(inputs[1])
    file_stats = pd.read_csv(inputs[2])
    log = read_transform_log(inputs[3])
    profile = profile_observations(obs, streams, file_stats, log, configured_intervals, gap_factor)

    out.mkdir(parents=True, exist_ok=True)
    manifest = RunManifest("profile", base_dir)
    for p in inputs:
        manifest.add_input(p)
    manifest.set_source({"m1_dir": manifest._rel(m1), "parameters": profile.parameters})
    tables = {"stream_profile": profile.stream_profile, "daily_profile": profile.daily_profile,
              "gaps": profile.gaps, "dataset_summary": profile.dataset_summary}
    for name, df in tables.items():
        path = out / f"{name}.parquet"
        df.to_parquet(path, index=False)
        manifest.add_output(path, table=name, rows=len(df))
    report = render_report(profile, {"m1_dir": manifest._rel(m1), "git_commit": manifest.data["git_commit"],
                                     "m1_manifest_sha256": manifest.data["inputs"][-1]["sha256"]})
    (out / "profile_report.md").write_text(report, encoding="utf-8")
    manifest.add_output(out / "profile_report.md")
    manifest.write(out / "profile_manifest.json")
    return profile
