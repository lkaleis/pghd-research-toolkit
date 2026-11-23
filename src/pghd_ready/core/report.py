"""Human-readable Markdown report for an ingestion run.

The report summarises, in plain language, what an ingestion run did. Every
number is computed from the run's own outputs (tables, transformation log,
column resolution, file statistics, rejected records); nothing is estimated.
It complements, and does not replace, the machine-readable audit files.

Status semantics:

* ``success``: files were read and standardised with no rejected records, no
  flagged values, no missing optional columns and no padded short rows.
* ``warnings``: ingestion completed, but at least one of the above occurred.
* ``failed``: ingestion stopped with an error; no tables were written.

None of these statuses is a statement about data quality or fitness for a
particular analysis.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from .. import __version__
from .annotations import AnnotationRegistry, IssueCategory
from .concepts import domain_record_counts
from .provenance import TransformRecord
from .sampling import INTERPOLATED, NATIVE, UNKNOWN, origin_evidence_counts, origin_labels

CATEGORY_TITLES = {
    IssueCategory.DOCUMENTED_ISSUE.value: "Documented issues",
    IssueCategory.SUSPECTED_ANOMALY.value: "Suspected anomalies",
    IssueCategory.SEMANTIC_AMBIGUITY.value: "Semantic ambiguities",
    IssueCategory.ASSUMPTION_BASED.value: "Assumption-based processing",
}
CATEGORY_MEANING = {
    IssueCategory.DOCUMENTED_ISSUE.value: "the source documentation states these values are erroneous",
    IssueCategory.SUSPECTED_ANOMALY.value: "something looks wrong but is not documented as such",
    IssueCategory.SEMANTIC_AMBIGUITY.value: "the meaning of the field or value is unclear; it was not interpreted",
    IssueCategory.ASSUMPTION_BASED.value: "values were produced or selected under an explicit assumption",
}
REASON_TEXT = {
    "field_count_mismatch": "row has more fields than the header, so values cannot be assigned to columns",
    "empty_row": "every field in the row is empty",
    "missing_participant_id": "no participant identifier could be determined for the row",
    "missing_timestamp": "the timestamp cell is empty",
    "unparseable_timestamp": "the timestamp text does not match the declared format",
    "timestamp_dst_unresolvable": "local time is ambiguous or does not exist at a daylight-saving change",
    "removed_by_transform": "observation removed by a configured named transform",
}
STATUS_TEXT = {
    "success": "SUCCESS - all source records were read and standardised without warnings",
    "warnings": "COMPLETED WITH WARNINGS - outputs were written, but some records or values need review",
    "failed": "FAILED - ingestion stopped with an error; no standardised tables were written",
}
NOT_ASSESSED = [
    "completeness and coverage over time (e.g. share of expected readings present per participant and day)",
    "gaps, sampling regularity and interpolation detection",
    "duplicate or conflicting records",
    "plausibility of values against physiological or device ranges",
    "agreement between streams that measure the same concept",
    "temporal alignment between streams and events",
    "task-specific eligibility and predictive fitness",
]
OUTPUT_DESCRIPTIONS = [
    ("observations.parquet", "standardised time-stamped measurements"),
    ("events.parquet", "standardised events with original attribute text in raw_attributes"),
    ("participant_attributes.parquet", "standardised participant-level values"),
    ("streams.parquet", "one row per participant and stream with record counts"),
    ("rejected_records.parquet", "source records that could not be kept, with full original content and reason"),
    ("ingest_summary.json", "aggregate counts (machine-readable)"),
    ("_provenance/transform_log.jsonl", "every processing step with basis, parameters and counts"),
    ("_provenance/column_resolution.csv", "which source column fed which role in every file"),
    ("_provenance/file_stats.csv", "per-file line, record, rejection and unused-column counts"),
    ("_provenance/unselected_filter_values.csv", "row-filter values that no configuration selected, per file"),
    ("_provenance/stream_diagnostics.csv", "method, evidence and confidence recorded by transforms, per participant and stream"),
    ("_provenance/manifest.json", "input/output hashes, configuration hashes, versions"),
]


# =========================================================================== context
@dataclass
class ReportContext:
    out_dir: Path
    base_dir: Path
    source: str | None = None
    config_version: str | None = None
    config_path: str | None = None
    data_root: str | None = None
    output_timezone: str | None = None
    date_shifted: bool | None = None
    temporal_notes: str | None = None
    generated_at: str = ""
    toolkit_version: str = __version__
    git_commit: str | None = None
    error: str | None = None
    config: Any = None
    registry: AnnotationRegistry | None = None
    file_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    resolution: pd.DataFrame = field(default_factory=pd.DataFrame)
    log: list[TransformRecord] = field(default_factory=list)
    rejected: pd.DataFrame = field(default_factory=pd.DataFrame)
    unselected: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: pd.DataFrame = field(default_factory=pd.DataFrame)
    concepts: Any = None
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    outputs_written: bool = False

    def rel(self, path: str | Path) -> str:
        p = Path(path).resolve()
        try:
            return p.relative_to(self.base_dir.resolve()).as_posix()
        except ValueError:
            return p.as_posix()

    @classmethod
    def from_result(cls, result, out_dir: Path, base_dir: Path, manifest: dict | None = None,
                    config_path: str | Path | None = None) -> "ReportContext":
        cfg = result.config
        ctx = cls(out_dir=Path(out_dir), base_dir=Path(base_dir), source=cfg.source, config_version=cfg.version,
                  output_timezone=cfg.temporal.output_timezone, date_shifted=cfg.temporal.date_shifted,
                  temporal_notes=cfg.temporal.notes, config=cfg, registry=result.registry,
                  file_stats=result.file_stats, resolution=result.column_resolution, log=result.transform_log.records,
                  rejected=result.rejected_records, unselected=result.unselected_values,
                  diagnostics=result.stream_diagnostics, concepts=result.concepts, outputs_written=True,
                  tables={"observations": result.observations, "events": result.events,
                          "participant_attributes": result.participant_attributes, "streams": result.streams})
        ctx.data_root = ctx.rel(result.data_root)
        ctx.config_path = ctx.rel(config_path) if config_path else None
        manifest = manifest or {}
        ctx.generated_at = manifest.get("created_at") or datetime.now(timezone.utc).isoformat()
        ctx.git_commit = manifest.get("git_commit")
        return ctx

    @classmethod
    def from_failure(cls, exc: BaseException, engine, config, out_dir: Path, base_dir: Path, data_root: Path,
                     config_path: str | Path | None = None) -> "ReportContext":
        ctx = cls(out_dir=Path(out_dir), base_dir=Path(base_dir), error=f"{type(exc).__name__}: {exc}",
                  generated_at=datetime.now(timezone.utc).isoformat())
        ctx.data_root = ctx.rel(data_root)
        ctx.config_path = ctx.rel(config_path) if config_path else None
        cfg = getattr(engine, "cfg", None) or (config if hasattr(config, "source") else None)
        if cfg is not None:
            ctx.config, ctx.source, ctx.config_version = cfg, cfg.source, cfg.version
            ctx.output_timezone, ctx.date_shifted = cfg.temporal.output_timezone, cfg.temporal.date_shifted
            ctx.temporal_notes = cfg.temporal.notes
        if engine is not None:
            ctx.registry = engine.registry
            try:
                engine._flush_steps()
            except Exception:  # partial counters may be inconsistent; the error is what matters
                pass
            ctx.log = list(engine.log.records)
            ctx.file_stats = pd.DataFrame(engine.file_stats)
            ctx.resolution = engine.resolution_frame()
            ctx.rejected = pd.DataFrame(engine.rejected)
            ctx.unselected = engine.unselected_frame()
            ctx.diagnostics = pd.DataFrame(engine.diagnostics)
            ctx.concepts = engine.concepts
        return ctx


# =========================================================================== status
def _flag_counts(ctx: ReportContext) -> dict[str, Counter]:
    """flag code -> Counter(table -> number of records carrying it)."""
    out: dict[str, Counter] = defaultdict(Counter)
    for name, df in ctx.tables.items():
        if "quality_flags" not in df or not len(df):
            continue
        for flags in df["quality_flags"]:
            for code in flags:
                out[code][name] += 1
    return out


def _missing_optional(ctx: ReportContext) -> pd.DataFrame:
    r = ctx.resolution
    if not len(r):
        return r
    return r[(r["status"] == "missing_optional") & ~r["role"].str.startswith("drop:")]


def determine_status(ctx: ReportContext) -> tuple[str, list[str]]:
    if ctx.error:
        return "failed", [ctx.error]
    reasons = []
    n_rej = len(ctx.rejected)
    if n_rej:
        reasons.append(f"{n_rej:,} source record(s) could not be kept (see 'Records not processed')")
    flagged = _flag_counts(ctx)
    n_flagged = sum(sum(c.values()) for c in flagged.values())
    if n_flagged:
        reasons.append(f"{n_flagged:,} flag(s) were attached to records (see 'Issues, ambiguities and assumptions')")
    miss = _missing_optional(ctx)
    if len(miss):
        reasons.append(f"{miss['role'].nunique()} optional field(s) were absent in at least one file")
    padded = int(ctx.file_stats["n_short_rows_padded"].sum()) if len(ctx.file_stats) else 0
    if padded:
        reasons.append(f"{padded:,} row(s) had fewer fields than the header and were padded with empty values")
    return ("warnings" if reasons else "success"), reasons


# =========================================================================== rendering helpers
def _table(headers: list[str], rows: list[list[Any]]) -> str:
    if not rows:
        return "_None._\n"

    def cell(v: Any) -> str:
        s = "" if v is None else str(v)
        return s.replace("|", "\\|").replace("\n", " ")

    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    lines += ["| " + " | ".join(cell(v) for v in r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def _n(x) -> str:
    return f"{int(x):,}"


def _role_target(ctx: ReportContext, role: str) -> tuple[str, str]:
    """Describe a column-resolution role as (standardised target, kind/details)."""
    cfg = ctx.config
    if role.endswith(".where"):
        owner = role[: -len(".where")].replace("observation:", "stream ").replace("event:", "event ")
        return f"row filter for {owner}", "row filter"
    if role.startswith("observation:"):
        stream = role.split(":", 1)[1]
        for t in getattr(cfg, "tables", ()):
            for o in t.observations:
                if o.stream_id == stream:
                    return f"stream `{stream}` -> concept `{o.concept}`", f"observation (source unit: {o.unit or 'none'})"
        return f"stream `{stream}`", "observation"
    if role.startswith("attribute:"):
        rest = role.split(":", 1)[1]
        if rest.endswith(".measured_at_raw"):
            return f"concept `{rest[:-len('.measured_at_raw')]}` (measurement time text)", "attribute metadata"
        unit = None
        for t in getattr(cfg, "tables", ()):
            for a in t.attributes:
                if a.concept == rest:
                    unit = a.unit
        return f"concept `{rest}`", f"participant attribute (source unit: {unit or 'none'})"
    if role.startswith("event:"):
        etype, part = role.split(":", 1)[1].split(".", 1)
        if part.startswith("attribute:"):
            return f"event `{etype}` attribute `{part.split(':', 1)[1]}`", "event attribute"
        return f"event `{etype}` ({part.replace('_', ' ')})", "event"
    return role.replace("_", " "), "structural"


def _origin_method(ctx: ReportContext, stream: str) -> str:
    """Describe how a stream's sampling origin was determined, from the configuration and the log."""
    parts = []
    for r in ctx.log:
        if r.target != stream:
            continue
        just = r.evidence or ", ".join(r.assumption_ids)
        if r.operation == "sampling_origin":
            parts.append(f"declared {r.params.get('declared_origin')} ({r.basis.value}: {just})")
        elif r.operation == "transform":
            parts.append(f"classified by `{r.function}` ({r.basis.value}: {just})")
    return "; ".join(parts) or "not declared or classified - unknown by default"


# =========================================================================== sections
def render_report(ctx: ReportContext) -> str:
    status, reasons = determine_status(ctx)
    out: list[str] = []
    w = out.append
    title = ctx.source or "unknown source"
    w(f"# Ingestion report: `{title}`\n")
    w(f"Generated {ctx.generated_at} by pghd_ready {ctx.toolkit_version}"
      + (f" (git {ctx.git_commit})" if ctx.git_commit else "") + ".\n")

    # 1. outcome ------------------------------------------------------------------
    w("## 1. Outcome\n")
    w(f"**{STATUS_TEXT[status]}.**\n")
    if status == "failed":
        w("Error:\n\n```\n" + (ctx.error or "") + "\n```\n")
    elif reasons:
        w("Reasons for warnings:\n")
        w("\n".join(f"- {r}" for r in reasons) + "\n")
    w("> This status only describes whether the source files could be read and converted into the standardised "
      "tables. It is **not** an assessment of data quality, completeness or fitness for any analysis "
      "(see section 12).\n")

    # 2. source -------------------------------------------------------------------
    w("## 2. Source and run settings\n")
    w(_table(["Setting", "Value"], [
        ["Source", ctx.source], ["Configuration", ctx.config_path or "(passed programmatically)"],
        ["Configuration version", ctx.config_version], ["Data root", ctx.data_root],
        ["Output timezone", ctx.output_timezone or "none (naive local clock time)"],
        ["Dates shifted for de-identification", {True: "yes - calendar dates are not real", False: "no",
                                                 None: "unknown"}[ctx.date_shifted]],
        ["Temporal notes", ctx.temporal_notes or ""],
    ]))

    fs = ctx.file_stats
    # 3. files -------------------------------------------------------------------
    w("## 3. Files and source records read\n")
    if len(fs):
        rows = []
        for tname, g in fs.groupby("table", sort=False):
            n_rec, n_rej = int(g["n_records"].sum()), int(g["n_rejected"].sum())
            rows.append([tname, _n(len(g)), _n(g["n_physical_lines"].sum()), _n(n_rec), _n(n_rec - n_rej), _n(n_rej),
                         _n(g["n_blank_lines"].sum()), _n(g["n_short_rows_padded"].sum())])
        w(_table(["Table", "Files", "Physical lines", "Data records", "Records used", "Records rejected",
                  "Blank lines skipped", "Short rows padded"], rows))
        w("Records are counted per source row. One source row can produce several standardised records "
          "(for example several streams measured at the same time).\n")
        w("<details><summary>Files processed</summary>\n")
        w(_table(["Table", "File", "Data records", "Rejected", "Participants"],
                 [[r.table, r.source_file, _n(r.n_records), _n(r.n_rejected), _n(r.n_participants)]
                  for r in fs.itertuples()]))
        w("</details>\n")
    else:
        w("_No files were read before the run stopped._\n")

    # 4. participants and outputs ----------------------------------------------------
    w("## 4. Participants and standardised records\n")
    if ctx.tables:
        all_p = set()
        rows = []
        for name in ("observations", "events", "participant_attributes", "streams"):
            df = ctx.tables[name]
            ps = set(df["participant_id"].dropna()) if len(df) else set()
            all_p |= ps
            rows.append([name, _n(len(df)), _n(len(ps))])
        rows.append(["rejected_records (audit)", _n(len(ctx.rejected)),
                     _n(ctx.rejected["participant_id"].dropna().nunique()) if len(ctx.rejected) else "0"])
        w(f"Participants with at least one standardised record: **{len(all_p):,}**.\n")
        w(_table(["Output table", "Records", "Participants"], rows))
        obs = ctx.tables["observations"]
        if len(obs):
            w("Observations by stream:\n")
            g = obs.groupby(["stream_id", "concept"], sort=True)
            w(_table(["Stream", "Concept", "Unit", "Records", "Participants", "Value empty (raw kept)"],
                     [[s, c, ", ".join(sorted(map(str, x["unit"].dropna().unique()))) or "none", _n(len(x)),
                       _n(x["participant_id"].nunique()), _n(x["value"].isna().sum())] for (s, c), x in g]))
        if len(obs):
            w("**Sampling origin of observations** (`is_interpolated`: native = False, interpolated = True, "
              "unknown = not declared or not classifiable). Unknown is the default; an origin is only native or "
              "interpolated if the configuration declared it with a basis or a classification transform inferred it "
              "(inferred classifications are assumption-based, not verified facts):\n")
            counts = origin_evidence_counts(obs, ctx.log).set_index("stream_id")
            w("Declared origins come from a configuration-level statement with a documented or assumption basis; "
              "inferred origins come from a row-by-row classification transform. Both are admitted by "
              "`native_only`, but they differ in evidence strength and should be distinguished in analyses:\n")
            rows = []
            for s, r in counts.iterrows():
                row = [s, _n(r["native_declared"]), _n(r["native_inferred"]), _n(r["interpolated_declared"]),
                       _n(r["interpolated_inferred"]), _n(r[UNKNOWN]), _origin_method(ctx, s)]
                if r["unattributed"]:
                    row[-1] += f"; {int(r['unattributed']):,} reading(s) with an origin but no recorded basis"
                rows.append(row)
            w(_table(["Stream", "Native (declared)", "Native (inferred)", "Interpolated (declared)",
                      "Interpolated (inferred)", "Unknown", "How the origin was determined"], rows))
        ev = ctx.tables["events"]
        if len(ev):
            w("Events by type and subtype:\n")
            g = ev.groupby(["event_type", ev["event_subtype"].fillna("(none)")], sort=True)
            w(_table(["Event type", "Subtype", "Records", "Participants"],
                     [[t, s, _n(len(x)), _n(x["participant_id"].nunique())] for (t, s), x in g]))
        att = ctx.tables["participant_attributes"]
        if len(att):
            w("Participant attributes by concept:\n")
            w(_table(["Concept", "Unit", "Records", "Value empty (raw kept)"],
                     [[c, ", ".join(sorted(map(str, x["unit"].dropna().unique()))) or "none", _n(len(x)),
                       _n((x["value_num"].isna() & x["value_text"].isna()).sum())]
                      for c, x in att.groupby("concept", sort=True)]))
        if ctx.concepts is not None and ctx.config is not None:
            dom = pd.DataFrame(domain_record_counts(ctx.concepts, obs, att, ev, ctx.config.event_domains()))
            if len(dom):
                w("**PGHD domains represented** (descriptive tags from the concept registries and event "
                  "configuration; a record counts once under each of its domains, so rows do not sum to the totals):\n")
                rows = []
                for d, g in dom.groupby(dom["domain"].fillna("(untagged)"), sort=True):
                    per = g.groupby("table")["n"].sum()
                    items = sorted(set(g.get("concept", pd.Series(dtype=object)).dropna())
                                   | set(g.get("event_type", pd.Series(dtype=object)).dropna()))
                    rows.append([d, _n(per.get("observations", 0)), _n(per.get("events", 0)),
                                 _n(per.get("participant_attributes", 0)), ", ".join(f"`{i}`" for i in items)])
                w(_table(["Domain", "Observations", "Events", "Attributes", "Concepts / event types"], rows))
    else:
        w("_No standardised tables were produced._\n")

    # 5. mappings ------------------------------------------------------------------
    w("## 5. Column mappings\n")
    res = ctx.resolution
    if len(res):
        rows = []
        for (tname, role), g in res[~res["role"].str.startswith("drop:")].groupby(["table", "role"], sort=False):
            target, kind = _role_target(ctx, role)
            matched = g["matched_column"].dropna()
            cols = ", ".join(f"`{c}`" for c in sorted(matched.unique())) or "(not found)"
            rows.append([tname, target, kind, cols, _n((g["status"] == "matched").sum()),
                         _n((g["status"] != "matched").sum())])
        w(_table(["Table", "Standardised target", "Kind", "Source column(s) used", "Files matched",
                  "Files missing"], rows))
        if len(fs):
            unused, dropped = Counter(), Counter()
            for r in fs.itertuples():
                unused.update((r.table, c) for c in (r.unused_columns or "").split("|") if c)
                dropped.update((r.table, c) for c in (r.dropped_columns or "").split("|") if c)
            w("Source columns present in the files but not referenced by the configuration. Their content is not "
              "carried into the standardised tables (it remains available in the source files):\n")
            w(_table(["Table", "Column", "Files"], [[t, f"`{c}`", _n(n)] for (t, c), n in sorted(unused.items())]))
            if dropped:
                w("Columns dropped on purpose by the configuration (`read.drop_columns`):\n")
                w(_table(["Table", "Column", "Files"], [[t, f"`{c}`", _n(n)] for (t, c), n in sorted(dropped.items())]))
    else:
        w("_No column resolution was recorded._\n")

    # 6. transformations -------------------------------------------------------------
    w("## 6. Value transformations\n")
    by_op: dict[str, list[TransformRecord]] = defaultdict(list)
    for r in ctx.log:
        by_op[r.operation or "other"].append(r)
    w("**Unit conversions** (multiplicative factors defined by the unit table or concept registry):\n")
    w(_table(["Table", "Target", "From", "To", "Factor", "Values converted"],
             [[r.table, r.target, r.params.get("from_unit"), r.params.get("to_unit"), r.params.get("factor"),
               _n(r.rows_out)] for r in by_op.get("unit_conversion", [])]))
    w("**Scaling of source encodings:**\n")
    w(_table(["Table", "Target", "Factor", "Basis", "Justification", "Values scaled"],
             [[r.table, r.target, r.params.get("factor"), r.basis.value, r.evidence or ", ".join(r.assumption_ids),
               _n(r.rows_out)] for r in by_op.get("scale", [])]))
    w("**Timezone localisation:**\n")
    w(_table(["Table", "Assumed zone", "Output zone", "Basis", "Justification", "Times localised", "Unresolvable"],
             [[r.table, r.params.get("assume_timezone"), r.params.get("output_timezone"), r.basis.value,
               r.evidence or ", ".join(r.assumption_ids), _n(r.rows_out), _n(r.rows_in - r.rows_out)]
              for r in by_op.get("timestamps.localize", [])]))
    w("**Named transforms** (plugin code declared in the configuration):\n")
    w(_table(["Table", "Stream", "Transform", "Basis", "Justification", "Rows in", "Rows out", "Flags raised"],
             [[r.table, r.target, r.function, r.basis.value, r.evidence or ", ".join(r.assumption_ids), _n(r.rows_in),
               _n(r.rows_out), ", ".join(f"{k}: {v:,}" for k, v in r.flags_raised.items()) or "none"]
              for r in by_op.get("transform", [])]))
    w("**Sampling-origin declarations** (stream-level, from the configuration):\n")
    w(_table(["Table", "Stream", "Declared origin", "Basis", "Justification", "Readings"],
             [[r.table, r.target, r.params.get("declared_origin"), r.basis.value,
               r.evidence or ", ".join(r.assumption_ids), _n(r.rows_out)] for r in by_op.get("sampling_origin", [])]))
    diag = ctx.diagnostics
    if len(diag):
        w("**Transform diagnostics** (method, evidence and confidence recorded by transforms per participant and "
          "stream; full detail in `_provenance/stream_diagnostics.csv`):\n")
        rows = []
        for (tr, s, k), g in diag.groupby(["transform", "stream_id", "key"], sort=True):
            vals = pd.to_numeric(g["value"], errors="coerce")
            if vals.notna().all():
                summary = f"min {vals.min():g}, median {vals.median():g}, max {vals.max():g}"
            else:
                uniq = sorted(map(str, g["value"].dropna().unique()))
                summary = ", ".join(uniq[:5]) + (f" (+{len(uniq) - 5} more)" if len(uniq) > 5 else "")
            rows.append([f"`{tr}`", s, k, _n(len(g)), summary])
        w(_table(["Transform", "Stream", "Diagnostic", "Participants", "Values"], rows))
    w("**Rules** (evaluated on the parsed source value, before scaling or unit conversion):\n")
    w(_table(["Table", "Target", "Condition", "Flag", "Category", "Action", "Records matched"],
             [[r.table, r.target, json.dumps(r.params.get("condition")), f"`{r.params.get('flag')}`",
               r.params.get("category"), r.params.get("action"), _n(sum(r.flags_raised.values()))]
              for r in by_op.get("rule", [])]))

    # 7. missing ----------------------------------------------------------------------
    w("## 7. Missing columns and expected data\n")
    miss = _missing_optional(ctx)
    if len(miss):
        rows = []
        for (tname, role), g in miss.groupby(["table", "role"], sort=False):
            files = sorted(g["file"])
            shown = ", ".join(f"`{f}`" for f in files[:5]) + (f" (+{len(files) - 5} more)" if len(files) > 5 else "")
            rows.append([tname, _role_target(ctx, role)[0], _n(len(files)), shown])
        w("Optional fields not found in some files. No values were invented; these files simply contribute no "
          "records for the field:\n")
        w(_table(["Table", "Standardised target", "Files missing", "Files"], rows))
    else:
        w("All configured optional fields were found in every file.\n")
    uns = ctx.unselected
    if len(uns):
        w("Values of row-filter columns that no configured observation or event selects. These rows were read but "
          "produced no standardised record for that filter (they are not errors, but check that the exclusion is "
          "intended, e.g. labels that differ only in spelling or case):\n")
        agg = uns.groupby(["table", "column", "value"], sort=True)["n_rows"].sum()
        w(_table(["Table", "Filter column", "Value not selected", "Rows"],
                 [[t, f"`{c}`", f"`{v}`", _n(n)] for (t, c, v), n in agg.items()]))
    if status == "failed" and len(res) and (res["status"] == "missing_required").any():
        w("A **required** column was missing, which stopped the run (see section 1).\n")

    # 8. values affected -------------------------------------------------------------
    w("## 8. Values that could not be used as-is\n")
    rows = []
    for r in by_op.get("parse_numeric", []):
        n = r.flags_raised.get("unparseable_numeric", 0)
        if n:
            rows.append(["Unparseable numeric value (original kept, standardised value empty)", r.table, r.target,
                         _n(n)])
    for r in by_op.get("rule", []):
        n = sum(r.flags_raised.values())
        if n and r.params.get("action") == "set_null":
            rows.append([f"Set to empty by documented-issue rule `{r.params.get('flag')}` (original kept)",
                         r.table, r.target, _n(n)])
    for r in by_op.get("map_category", []):
        n = sum(r.flags_raised.values())
        if n:
            rows.append(["Label not covered by the value map (raw label kept, no category)", r.table, r.target, _n(n)])
    for r in by_op.get("timestamps.parse", []):
        if r.rows_in - r.rows_out:
            rows.append(["Unparseable timestamp (row rejected, content kept)", r.table, "timestamp",
                         _n(r.rows_in - r.rows_out)])
    for r in by_op.get("timestamps.localize", []):
        if r.rows_in - r.rows_out:
            rows.append(["Timestamp unresolvable at daylight-saving change (row rejected, content kept)", r.table,
                         "timestamp", _n(r.rows_in - r.rows_out)])
    for r in by_op.get("transform", []):
        if r.rows_in - r.rows_out > 0:
            rows.append([f"Removed by transform `{r.function}` (kept in rejected_records)", r.table, r.target,
                         _n(r.rows_in - r.rows_out)])
    w(_table(["What happened", "Table", "Target", "Count"], rows))

    # 9. annotations -----------------------------------------------------------------
    w("## 9. Issues, ambiguities and assumptions encountered\n")
    flagged = _flag_counts(ctx)
    reg = ctx.registry
    if flagged and reg is not None:
        for cat in IssueCategory:
            codes = [c for c in sorted(flagged) if reg.flags[c].category is cat]
            if not codes:
                continue
            w(f"**{CATEGORY_TITLES[cat.value]}** ({CATEGORY_MEANING[cat.value]}):\n")
            w(_table(["Flag", "Description", "Evidence / assumption", "Records flagged (by table)"],
                     [[f"`{c}`", reg.flags[c].description, reg.flags[c].evidence or reg.flags[c].assumption_id or "",
                       ", ".join(f"{t}: {n:,}" for t, n in sorted(flagged[c].items()))] for c in codes]))
    else:
        w("No flags were attached to any standardised record.\n")
    used_a: dict[str, list[str]] = defaultdict(list)
    for r in ctx.log:
        for a in r.assumption_ids:
            used_a[a].append(f"{r.operation} ({r.table}{'/' + r.target if r.target else ''})")
    if used_a and reg is not None:
        w("**Assumptions applied in this run** (results depend on them being true):\n")
        w(_table(["Assumption", "Statement", "Rationale", "Used by"],
                 [[a, reg.assumptions[a].statement, reg.assumptions[a].rationale, "; ".join(sorted(set(v)))]
                  for a, v in sorted(used_a.items()) if a in reg.assumptions]))
    else:
        w("No registered assumptions were applied in this run.\n")

    # 10. rejected --------------------------------------------------------------------
    w("## 10. Records not processed\n")
    rej = ctx.rejected
    if len(rej):
        w(f"{len(rej):,} record(s) were not converted into standardised records. Their complete original content and "
          "the reason are stored in `rejected_records.parquet` (`raw_record` column).\n")
        w(_table(["Table", "Reason", "Meaning", "Records"],
                 [[t, f"`{r}`", REASON_TEXT.get(r, ""), _n(len(g))]
                  for (t, r), g in rej.groupby(["table", "reason"], sort=True)]))
        ex = rej.sort_values(["source_file", "source_row"]).head(15)
        w("First records (file and physical line number; the header is line 1):\n")
        w(_table(["File", "Line", "Reason", "Detail"],
                 [[f"`{r.source_file}`", "" if pd.isna(r.source_row) else int(r.source_row), r.reason, r.detail or ""]
                  for r in ex.itertuples()]))
        if len(rej) > 15:
            w(f"... and {len(rej) - 15:,} more in `rejected_records.parquet`.\n")
    else:
        w("Every source record was used." if status != "failed" else "No rejection information is available.")
        w("\n")

    # 11. outputs ---------------------------------------------------------------------
    w("## 11. Where the outputs are\n")
    if ctx.outputs_written:
        w(_table(["File", "Content"], [[f"`{ctx.rel(ctx.out_dir / f)}`", d] for f, d in OUTPUT_DESCRIPTIONS]))
        w("`source_row` in every table is the 1-based physical line in `source_file` where the record starts "
          "(header = line 1), so a record can be found with any text editor.\n")
    else:
        w(f"No tables or audit files were written. This report is at `{ctx.rel(ctx.out_dir / 'ingestion_report.md')}`.\n")

    # 12. not assessed ----------------------------------------------------------------
    w("## 12. What this run did not assess\n")
    w("Ingestion only standardises and annotates the source. The following are **not implemented yet** and were "
      "therefore not evaluated (planned for the quality-assessment and task-fitness milestones):\n")
    w("\n".join(f"- {x}" for x in NOT_ASSESSED) + "\n")

    # 13. interpretation --------------------------------------------------------------
    w("## 13. Interpretation for analysts\n")
    w("\n".join(f"- {b}" for b in _interpretation(ctx, status, flagged)) + "\n")
    return "\n".join(out)


def _interpretation(ctx: ReportContext, status: str, flagged: dict[str, Counter]) -> list[str]:
    b: list[str] = []
    if status == "failed":
        b.append("The run failed, so no standardised data exists for this configuration and input. Fix the cause "
                 "in section 1 and run again; nothing from this run should be analysed.")
        return b
    fs, rej = ctx.file_stats, ctx.rejected
    n_rec = int(fs["n_records"].sum()) if len(fs) else 0
    n_src_rej = int((rej["record_scope"] == "source_row").sum()) if len(rej) else 0
    if n_rec:
        b.append(f"{n_rec - n_src_rej:,} of {n_rec:,} source records ({100 * (n_rec - n_src_rej) / n_rec:.1f}%) were "
                 "used. This measures how much of the files could be read, not how complete the data is.")
    if n_src_rej:
        reasons = ", ".join(f"{r} ({n:,})" for r, n in rej[rej["record_scope"] == "source_row"]
                            .groupby("reason").size().items())
        b.append(f"Review the rejected records ({reasons}) in `rejected_records.parquet` before deciding whether "
                 "they can be recovered (e.g. by correcting the configuration) or are genuinely unusable.")
    reg = ctx.registry
    if reg is not None and flagged:
        by_cat: Counter = Counter()
        for code, c in flagged.items():
            by_cat[reg.flags[code].category.value] += sum(c.values())
        if by_cat.get("documented_issue"):
            b.append(f"{by_cat['documented_issue']:,} record(s) carry documented issues. Where a rule emptied the value, "
                     "the original is still in `raw_value` / `raw_attributes`; do not restore it without justification.")
        if by_cat.get("suspected_anomaly"):
            b.append(f"{by_cat['suspected_anomaly']:,} record(s) carry suspected anomalies (for example values that "
                     "could not be parsed). Decide how they should be treated; ingestion did not guess.")
        if by_cat.get("semantic_ambiguity"):
            codes = sorted(c for c in flagged if reg.flags[c].category is IssueCategory.SEMANTIC_AMBIGUITY)
            b.append(f"{by_cat['semantic_ambiguity']:,} record(s) have unresolved meaning ({', '.join(codes)}). Do not "
                     "use these fields or labels as analysis variables until their meaning is confirmed.")
        if by_cat.get("assumption_based"):
            b.append(f"{by_cat['assumption_based']:,} record(s) depend on explicit assumptions; conclusions inherit "
                     "them (see section 9).")
    if any(r.assumption_ids for r in ctx.log if r.operation == "timestamps.localize"):
        b.append("Clock times were converted using an assumed timezone. If that assumption is wrong, every derived "
                 "time (and any alignment between sources) is shifted.")
    miss = _missing_optional(ctx)
    if len(miss):
        b.append(f"{miss['role'].nunique()} optional field(s) are missing in some files, so the available variables "
                 "differ between participants. Check whether this restricts the planned analysis.")
    obs = ctx.tables.get("observations")
    if obs is not None and len(obs):
        labels = origin_labels(obs["is_interpolated"]).value_counts()
        n_unk, n_int = int(labels.get(UNKNOWN, 0)), int(labels.get(INTERPOLATED, 0))
        if n_unk or n_int:
            b.append(f"Sampling origin: {int(labels.get(NATIVE, 0)):,} observation(s) are native, {n_int:,} "
                     f"interpolated and {n_unk:,} of unknown origin (section 4). Downstream analyses must choose an "
                     "explicit inclusion policy (native only, native and unknown, or all) and should report "
                     "sensitivity across policies; interpolated values are not measurements.")
    if len(ctx.unselected):
        vals = ", ".join(sorted(f"'{v}'" for v in ctx.unselected["value"].unique()))
        b.append(f"Some rows were not selected by any row filter (values {vals}; section 7). Confirm these "
                 "exclusions are intended and not spelling or case variants of selected values.")
    if ctx.date_shifted:
        b.append("Dates were shifted for de-identification: calendar dates, weekdays and seasons are not meaningful.")
    b.append("Successful ingestion does not mean the data are suitable for analysis. Coverage, gaps, duplicates, "
             "plausibility, cross-stream agreement and alignment have not been assessed yet (section 12).")
    return b
