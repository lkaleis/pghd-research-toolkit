"""Generic, configuration-driven ingestion of tabular (delimited text) sources.

:func:`ingest_source` reads the files declared in a :class:`SourceConfig` and
produces the common-representation tables (observations, events,
participant_attributes, streams) plus audit tables (rejected_records,
file_stats, column_resolution) and a transformation log.
:func:`ingest_and_write` additionally validates and writes everything with a
run manifest and a human-readable ``ingestion_report.md`` (also on failure).

Principles enforced here:

* All source values are read as text; the original token is kept in
  ``raw_value`` (observations, attributes) or ``raw_attributes`` (events).
* ``source_row`` is the physical line where the record starts (header = 1).
* Columns are resolved by alias; a missing required column or an ambiguous
  match is an error; a missing optional column is recorded, never guessed.
* Source rows that cannot be used (no participant ID, missing/unparseable
  timestamp, too many fields, ...) are kept in ``rejected_records`` with their
  full original content and the reason.
* Values are only changed by declared steps (rules, scale, unit conversion,
  timezone localisation, named transforms), each logged with its basis.
* Named transforms may change values only on rows they mark with a declared,
  explanatory flag; rows they remove are kept in ``rejected_records``.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from . import units as _units
from .annotations import AnnotationRegistry, FlagAction, IssueCategory, validate_flags
from .concepts import ConceptKind, ConceptRegistry, ValueType, check_table_concepts
from .config import (
    AttributeSpec,
    ConfigError,
    EventSpec,
    ObservationSpec,
    RuleSpec,
    SourceConfig,
    TableSpec,
    TransformSpec,
    WhereSpec,
    normalize_header,
)
from .io import write_table
from .provenance import RunManifest, TransformBasis, TransformLog, _utc_now, canonical_json
from .schemas import (
    event_schema,
    observation_schema,
    participant_attribute_schema,
    rejected_record_schema,
    stream_schema,
)
from .transforms import TransformContext, get_transform, load_plugins, resolve_callable


class IngestError(RuntimeError):
    pass


BUILTIN_ANNOTATIONS = AnnotationRegistry.from_mapping(
    {
        "flags": {
            "unparseable_numeric": {
                "category": "suspected_anomaly",
                "description": "Source value could not be parsed as a number; raw value kept, standardised value empty.",
            },
            "unmapped_category": {
                "category": "semantic_ambiguity",
                "description": "Source label is not covered by the configured value map; raw label kept, no category assigned.",
            },
        }
    }
)

OBS_COLUMNS = ["participant_id", "source_dataset", "stream_id", "concept", "timestamp", "value", "unit", "raw_value",
               "is_interpolated", "quality_flags", "source_file", "source_row", "source_column"]
EVENT_CORE_COLUMNS = ["participant_id", "source_dataset", "event_id", "event_type", "event_subtype", "raw_label",
                      "start_time", "end_time", "raw_attributes", "quality_flags", "source_file", "source_row"]
ATTR_COLUMNS = ["participant_id", "source_dataset", "concept", "value_num", "value_text", "unit", "raw_value",
                "measured_at", "measured_at_raw", "quality_flags", "source_file", "source_row", "source_column"]
STREAM_COLUMNS = ["participant_id", "source_dataset", "stream_id", "concept", "device", "declared_native_interval_s",
                  "first_timestamp", "last_timestamp", "n_rows_source", "n_rows_output", "quality_flags"]
REJECT_COLUMNS = ["source_dataset", "table", "source_file", "source_row", "record_scope", "participant_id", "reason",
                  "detail", "raw_record"]
FILE_STAT_COLUMNS = ["table", "source_file", "n_physical_lines", "n_records", "n_blank_lines", "n_short_rows_padded",
                     "n_rejected", "n_participants", "dropped_columns", "unused_columns"]
# observation columns a transform must never modify on rows it keeps
_PROTECTED = ["participant_id", "source_dataset", "stream_id", "concept", "unit", "raw_value", "source_column"]

_OFFSET_RE = re.compile(r"(?:Z|[+-]\d{2}:?\d{2})$")


# =========================================================================== result
@dataclass
class IngestResult:
    config: SourceConfig
    concepts: ConceptRegistry
    registry: AnnotationRegistry
    observations: pd.DataFrame
    events: pd.DataFrame
    participant_attributes: pd.DataFrame
    streams: pd.DataFrame
    rejected_records: pd.DataFrame
    file_stats: pd.DataFrame
    column_resolution: pd.DataFrame
    unselected_values: pd.DataFrame
    transform_log: TransformLog
    input_files: list[Path]
    data_root: Path
    event_schema_fn: Callable

    def summary(self) -> dict[str, Any]:
        """Aggregate-only description of the run (no record-level content)."""

        def flag_summary(df: pd.DataFrame) -> list[dict]:
            c: Counter = Counter()
            for flags in df["quality_flags"]:
                c.update(flags)
            return [{"code": k, "category": self.registry.category_of(k).value, "n": v} for k, v in sorted(c.items())]

        obs, ev, att, rej = self.observations, self.events, self.participant_attributes, self.rejected_records
        res, fs = self.column_resolution, self.file_stats
        return {
            "source": self.config.source,
            "config_version": self.config.version,
            "n_input_files": len(self.input_files),
            "source_records": int(fs["n_records"].sum()) if len(fs) else 0,
            "participants": {
                "observations": int(obs["participant_id"].nunique()),
                "events": int(ev["participant_id"].nunique()),
                "participant_attributes": int(att["participant_id"].nunique()),
            },
            "rows": {"observations": len(obs), "events": len(ev), "participant_attributes": len(att),
                     "streams": len(self.streams), "rejected_records": len(rej)},
            "rejected_by_reason": {k: int(v) for k, v in rej.groupby("reason").size().items()} if len(rej) else {},
            "observations_by_stream": {k: int(v) for k, v in obs.groupby("stream_id").size().items()},
            "events_by_type_subtype": [
                {"event_type": t, "event_subtype": s, "n": int(n)}
                for (t, s), n in ev.groupby(["event_type", ev["event_subtype"].fillna("<none>")]).size().items()
            ],
            "attributes_by_concept": {k: int(v) for k, v in att.groupby("concept").size().items()},
            "flags": {"observations": flag_summary(obs), "events": flag_summary(ev),
                      "participant_attributes": flag_summary(att)},
            "column_resolution": [
                {"table": t, "role": r, "status": s, "n_files": int(n)}
                for (t, r, s), n in res.groupby(["table", "role", "status"]).size().items()
            ] if len(res) else [],
            "transform_steps": len(self.transform_log.records),
        }


# =========================================================================== helpers
@dataclass
class _Step:
    """Aggregated counters for one logged processing step (flushed into the TransformLog)."""

    step: str
    operation: str
    table: str | None
    target: str | None
    function: str = "core.ingest"
    basis: TransformBasis = TransformBasis.STRUCTURAL
    evidence: str | None = None
    assumption_ids: tuple[str, ...] = ()
    rows_in: int = 0
    rows_out: int = 0
    flags: Counter = field(default_factory=Counter)
    params: dict[str, set] = field(default_factory=dict)
    notes: set[str] = field(default_factory=set)
    fixed_params: dict[str, Any] = field(default_factory=dict)


@dataclass
class _ParsedFile:
    df: pd.DataFrame  # all values text or NaN, columns = (deduplicated) header names
    lines: pd.Series  # start line per record, aligned to df.index
    header_map: dict[str, str]
    n_physical_lines: int
    n_records: int  # data records found (including rejected over-long ones)
    n_blank_lines: int
    n_padded: int
    overlong: list[tuple[int, list[str]]]  # (line, fields) rows with more fields than the header


def _new_flags(n: int) -> list[list[str]]:
    return [[] for _ in range(n)]


def _add_flag(flags: list[list[str]], positions, code: str) -> int:
    count = 0
    for i in positions:
        if code not in flags[i]:
            flags[i].append(code)
            count += 1
    return count


def _rule_mask(rule: RuleSpec, numeric: pd.Series | None, text: pd.Series) -> np.ndarray:
    op, x = rule.op, rule.operand
    if numeric is not None:
        s = numeric
        if op == "equals":
            m = s == float(x)
        elif op == "in":
            m = s.isin([float(v) for v in x])
        elif op == "below":
            m = s < float(x)
        elif op == "above":
            m = s > float(x)
        else:  # outside
            m = (s < float(x[0])) | (s > float(x[1]))
        return m.fillna(False).to_numpy(dtype=bool)
    if op not in ("equals", "in"):
        raise IngestError(f"Rule operator {op!r} needs numeric values.")
    vals = [str(x)] if op == "equals" else [str(v) for v in x]
    return text.isin(vals).to_numpy(dtype=bool)


def _file_key(rel: str) -> str:
    return hashlib.sha1(rel.encode("utf-8")).hexdigest()[:8]


def _empty(columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame({c: pd.Series(dtype=object) for c in columns})


def _dedupe_header(header: list[str]) -> list[str]:
    """Make duplicate header names unique the way pandas does ('Time', 'Time.1', ...)."""
    seen: dict[str, int] = {}
    out: list[str] = []
    for h in header:
        if h not in seen:
            seen[h] = 0
            out.append(h)
            continue
        k = seen[h]
        while True:
            k += 1
            cand = f"{h}.{k}"
            if cand not in seen:
                break
        seen[h] = k
        seen[cand] = 0
        out.append(cand)
    return out


def _json_row(row: pd.Series) -> str:
    return json.dumps({str(k): (None if pd.isna(v) else str(v)) for k, v in row.items()}, ensure_ascii=False)


def _same(a: pd.Series, b: pd.Series) -> pd.Series:
    return (a == b) | (a.isna() & b.isna())


# =========================================================================== engine
class _Engine:
    def __init__(self, config: SourceConfig, data_root: Path, concepts: ConceptRegistry, clock):
        self.cfg = config
        self.data_root = Path(data_root).resolve()
        self.concepts = concepts
        self.tz = config.temporal.output_timezone
        self.registry = BUILTIN_ANNOTATIONS.merge(config.annotations)
        self.log = TransformLog(self.registry, clock=clock)
        self.resolution: list[dict] = []
        self.file_stats: list[dict] = []
        self.rejected: list[dict] = []
        self.unselected: list[dict] = []  # filter-column values that no spec selects
        self.input_files: list[Path] = []
        self._steps: dict[str, _Step] = {}
        self._used_cols: set[str] = set()
        self._validate_static()

    # ------------------------------------------------------------- static validation
    def _validate_static(self) -> None:
        cfg = self.cfg
        unknown_flags = self.registry.unknown_codes(cfg.referenced_flags())
        if unknown_flags:
            raise ConfigError(f"Config uses unregistered flag codes {sorted(unknown_flags)}; define them under annotations.flags.")
        missing_a = cfg.referenced_assumptions() - set(self.registry.assumptions)
        if missing_a:
            raise ConfigError(f"Config references undefined assumptions {sorted(missing_a)}.")
        load_plugins(cfg.plugins)
        self.event_schema_fn = resolve_callable(cfg.event_schema) if cfg.event_schema else event_schema
        ev_schema = self.event_schema_fn(self.tz)
        self.event_attr_units = dict((ev_schema.metadata or {}).get("attribute_units", {}))
        self.event_columns = list(ev_schema.columns)
        self.event_float_columns = tuple(c for c, col in ev_schema.columns.items() if str(col.dtype).startswith("float"))
        missing_core = [c for c in EVENT_CORE_COLUMNS if c not in self.event_columns]
        if missing_core:
            raise ConfigError(f"Event schema lacks core columns {missing_core}.")
        for t in cfg.tables:
            for o in t.observations:
                c = self.concepts.get(o.concept)
                if c.kind is not ConceptKind.OBSERVATION:
                    raise ConfigError(f"Stream {o.stream_id}: concept {o.concept} is not an observation concept.")
                self._check_unit(f"stream {o.stream_id}", c, o.unit)
                for tr in o.transforms:
                    get_transform(tr.name)
                    self._check_transform_flags(o.stream_id, tr)
            for a in t.attributes:
                c = self.concepts.get(a.concept)
                if c.kind is not ConceptKind.ATTRIBUTE:
                    raise ConfigError(f"Attribute {a.concept} is not an attribute concept.")
                self._check_unit(f"attribute {a.concept}", c, a.unit)
            for e in t.events:
                for at in e.attributes:
                    if at.target not in self.event_columns or at.target in EVENT_CORE_COLUMNS:
                        raise ConfigError(f"Event attribute {at.target!r} is not an extension column of the event schema.")
                    target_unit = self.event_attr_units.get(at.target)
                    if at.kind == "numeric" and (target_unit is None) != (at.unit is None):
                        raise ConfigError(
                            f"Event attribute {at.target!r}: source unit must be declared exactly when the target has a "
                            f"unit (target unit: {target_unit!r})."
                        )
                    if at.unit and target_unit:
                        _units.conversion_factor(at.unit, target_unit)

    def _check_transform_flags(self, stream: str, tr: TransformSpec) -> None:
        j = tr.justification
        for code in tr.flags:
            fd = self.registry.flags[code]
            if fd.category is IssueCategory.ASSUMPTION_BASED and fd.assumption_id not in j.assumption_ids:
                raise ConfigError(f"Stream {stream}, transform {tr.name}: flag {code} relies on assumption "
                                  f"{fd.assumption_id}, which the transform does not declare in assumption_ids.")
            if fd.category is IssueCategory.DOCUMENTED_ISSUE and j.basis is not TransformBasis.DOCUMENTED:
                raise ConfigError(f"Stream {stream}, transform {tr.name}: documented-issue flag {code} needs basis "
                                  "'documented'.")

    def _explanatory_flags(self, tr: TransformSpec) -> set[str]:
        """Declared flags that may justify a changed value for this transform."""
        out = set()
        for code in tr.flags:
            fd = self.registry.flags[code]
            if fd.category is IssueCategory.ASSUMPTION_BASED and fd.assumption_id in tr.justification.assumption_ids:
                out.add(code)
            elif fd.category is IssueCategory.DOCUMENTED_ISSUE and tr.justification.basis is TransformBasis.DOCUMENTED:
                out.add(code)
        return out

    @staticmethod
    def _check_unit(where: str, concept, unit: str | None) -> None:
        if concept.value_type is ValueType.NUMERIC:
            if unit is None:
                raise ConfigError(f"{where}: numeric concept {concept.id} requires a source 'unit'.")
            concept.factor_to_canonical(unit)  # raises if undefined
        elif unit is not None:
            raise ConfigError(f"{where}: {concept.value_type.value} concept {concept.id} takes no unit.")
        if concept.value_type is ValueType.CATEGORICAL and concept.kind is ConceptKind.OBSERVATION:
            raise ConfigError(f"{where}: categorical observations are not supported yet.")

    # ------------------------------------------------------------- step logging
    def _step(self, key: str, operation: str, table: str | None, target: str | None, **meta) -> _Step:
        s = self._steps.get(key)
        if s is None:
            s = _Step(key, operation, table, target, **meta)
            self._steps[key] = s
        return s

    @staticmethod
    def _count(s: _Step, rows_in: int, rows_out: int, flags: dict | None = None, **params) -> None:
        s.rows_in += int(rows_in)
        s.rows_out += int(rows_out)
        if flags:
            s.flags.update({k: v for k, v in flags.items() if v})
        for k, v in params.items():
            s.params.setdefault(k, set()).add(v if not isinstance(v, list) else tuple(v))

    def _rule_step(self, key: str, table: str, target: str, rule: RuleSpec) -> _Step:
        fd = self.registry.flags[rule.flag]
        action = "value set to empty (original kept)" if fd.action is FlagAction.SET_NULL else "flag only"
        if fd.category is IssueCategory.DOCUMENTED_ISSUE:
            meta = {"basis": TransformBasis.DOCUMENTED, "evidence": fd.evidence}
        elif fd.category is IssueCategory.ASSUMPTION_BASED:
            meta = {"basis": TransformBasis.ASSUMPTION, "assumption_ids": (fd.assumption_id,)}
        else:
            meta = {}
        s = self._step(f"{key}.rule.{rule.flag}", "rule", table, target, function="core.ingest.rule", **meta)
        s.fixed_params.update({"condition": {rule.op: rule.operand}, "flag": rule.flag,
                               "category": fd.category.value, "action": fd.action.value})
        s.notes.add(f"evaluated on the parsed source value; {action}")
        return s

    def _flush_steps(self) -> None:
        for s in self._steps.values():
            params = {k: sorted(map(str, v)) if len(v) > 1 else str(next(iter(v))) for k, v in s.params.items()}
            params.update(s.fixed_params)
            self.log.record(s.step, s.function, s.basis, s.rows_in, s.rows_out, params=params,
                            assumption_ids=list(s.assumption_ids), evidence=s.evidence, flags_raised=dict(s.flags),
                            notes="; ".join(sorted(s.notes)) or None, operation=s.operation, table=s.table,
                            target=s.target)
        self._steps.clear()

    # ------------------------------------------------------------- reading
    def _files(self, t: TableSpec) -> list[Path]:
        if t.path:
            p = self.data_root / t.path
            if not p.is_file():
                raise IngestError(f"Table {t.name}: file {t.path} not found under data root.")
            return [p]
        files = sorted(p for p in self.data_root.glob(t.glob) if p.is_file())
        if not files:
            raise IngestError(f"Table {t.name}: no files match {t.glob!r}.")
        return files

    def _read(self, t: TableSpec, path: Path, rel: str) -> _ParsedFile:
        """Read a delimited file as text, tracking the physical start line of every record."""
        with open(path, encoding=t.read.encoding, newline="") as fh:
            reader = csv.reader(fh, delimiter=t.read.delimiter)
            try:
                header = next(reader)
            except StopIteration as exc:
                raise IngestError(f"{rel}: file is empty.") from exc
            names = _dedupe_header(header)
            records, lines, overlong = [], [], []
            blank = padded = 0
            next_start = reader.line_num + 1
            for rec in reader:
                start, next_start = next_start, reader.line_num + 1
                if not rec:
                    blank += 1
                    continue
                if len(rec) > len(names):
                    overlong.append((start, rec))
                    continue
                if len(rec) < len(names):
                    padded += 1
                    rec = rec + [""] * (len(names) - len(rec))
                records.append(rec)
                lines.append(start)
            n_lines = reader.line_num
        df = pd.DataFrame(records, columns=names, dtype=object)
        na = set(t.read.na_values)
        if len(df):
            df = df.mask(df.isin(na))
        header_map: dict[str, str] = {}
        for col in df.columns:
            norm = normalize_header(col)
            if norm in header_map:
                raise IngestError(f"{rel}: columns {header_map[norm]!r} and {col!r} are indistinguishable after normalisation.")
            header_map[norm] = col
        return _ParsedFile(df, pd.Series(lines, index=df.index, dtype="Int64"), header_map, n_lines,
                           len(records) + len(overlong), blank, padded, overlong)

    def _resolve(self, table: str, rel: str, role: str, aliases: tuple[str, ...], header_map: dict[str, str],
                 required: bool) -> str | None:
        found = list(dict.fromkeys(header_map[normalize_header(a)] for a in aliases if normalize_header(a) in header_map))
        if len(found) > 1:
            raise IngestError(f"{rel}: {role} matches several columns {found}; aliases must be unambiguous.")
        status = "matched" if found else ("missing_required" if required else "missing_optional")
        self.resolution.append({"table": table, "file": rel, "role": role, "aliases": "|".join(aliases),
                                "matched_column": found[0] if found else None, "status": status})
        if found:
            self._used_cols.add(found[0])
        if not found and required:
            raise IngestError(f"{rel}: required column for {role} not found (aliases {list(aliases)}).")
        return found[0] if found else None

    def _where(self, t: TableSpec, rel: str, role: str, df: pd.DataFrame, hm: dict, w: WhereSpec | None) -> pd.Series:
        if w is None:
            return pd.Series(True, index=df.index)
        col = self._resolve(t.name, rel, f"{role}.where", w.column, hm, required=True)
        return df[col].str.strip().isin(w.values)

    def _reject(self, t: TableSpec, rel: str, line, reason: str, raw_record: str, participant_id=None,
                detail: str | None = None, scope: str = "source_row") -> None:
        self.rejected.append({"source_dataset": self.cfg.source, "table": t.name, "source_file": rel,
                              "source_row": None if line is None or pd.isna(line) else int(line),
                              "record_scope": scope, "participant_id": participant_id, "reason": reason,
                              "detail": detail, "raw_record": raw_record})

    # ------------------------------------------------------------- timestamps
    def _parse_timestamps(self, t: TableSpec, rel: str, raw: pd.Series) -> tuple[pd.Series, pd.Series]:
        """Return (timestamps, reason) where reason names why a row has no usable timestamp."""
        spec = t.timestamp
        fmt = spec.format
        nonnull = raw.notna()
        reason = pd.Series(None, index=raw.index, dtype=object)
        reason[~nonnull] = "missing_timestamp"
        has_offset = raw[nonnull].str.strip().str.contains(_OFFSET_RE)
        if len(has_offset) and has_offset.all():
            # explicit offsets (possibly differing, e.g. across DST): exact conversion to UTC
            parsed = pd.to_datetime(raw, format=fmt, errors="coerce", utc=True)
        elif not has_offset.any():
            parsed = pd.to_datetime(raw, format=fmt, errors="coerce")
        else:
            raise IngestError(f"{rel}: timestamps mix values with and without UTC offsets.")
        reason[nonnull & parsed.isna()] = "unparseable_timestamp"
        s = self._step(f"{t.name}.timestamps.parse", "timestamps.parse", t.name, None)
        self._count(s, int(nonnull.sum()), int(parsed.notna().sum()), format=fmt)
        aware = isinstance(parsed.dtype, pd.DatetimeTZDtype)
        if aware:
            if spec.assume_timezone:
                raise IngestError(f"{rel}: assume_timezone given but timestamps already carry offsets.")
            if self.tz is None:
                raise IngestError(f"{rel}: timestamps carry offsets but output_timezone is null (naive); "
                                  "declare an output timezone.")
            out = parsed.dt.tz_convert(self.tz)
        elif self.tz is None:
            out = parsed
        else:
            if not spec.assume_timezone:
                raise IngestError(f"{rel}: naive timestamps but output_timezone={self.tz!r}; declare assume_timezone.")
            j = spec.timezone_justification
            localized = parsed.dt.tz_localize(spec.assume_timezone, ambiguous="NaT", nonexistent="NaT")
            lost = parsed.notna() & localized.isna()
            reason[lost] = "timestamp_dst_unresolvable"
            ls = self._step(f"{t.name}.timestamps.localize", "timestamps.localize", t.name, None,
                            function="core.ingest.localize", basis=j.basis, evidence=j.evidence,
                            assumption_ids=j.assumption_ids)
            ls.fixed_params.update({"assume_timezone": spec.assume_timezone, "output_timezone": self.tz})
            ls.notes.add("naive times localised to the declared zone; DST-ambiguous or nonexistent times are "
                         "rejected, never shifted")
            self._count(ls, int(parsed.notna().sum()), int(localized.notna().sum()))
            out = localized.dt.tz_convert(self.tz)
        return out.dt.as_unit("ns"), reason

    # ------------------------------------------------------------- observations
    def _observations(self, t: TableSpec, o: ObservationSpec, pf: _ParsedFile, rel, pid, ts, ok) -> pd.DataFrame | None:
        df, hm = pf.df, pf.header_map
        col = self._resolve(t.name, rel, f"observation:{o.stream_id}", o.column, hm, o.required)
        if col is None:
            return None
        where = self._where(t, rel, f"observation:{o.stream_id}", df, hm, o.where)
        mask = df[col].notna() & ok & where
        sub_idx = df.index[mask]
        raw = df.loc[sub_idx, col]
        n = len(raw)
        concept = self.concepts.get(o.concept)
        key = f"{t.name}.observations.{o.stream_id}"
        s = self._step(f"{key}.extract", "extract", t.name, o.stream_id)
        s.notes.add("rows with an empty cell, a rejected source row or a non-matching 'where' filter yield no record")
        self._count(s, int((ok & where).sum()), n, column=col)
        flags = _new_flags(n)
        value = pd.to_numeric(raw.str.strip(), errors="coerce").astype(float)
        bad = np.flatnonzero(value.isna().to_numpy())
        self._count(self._step(f"{key}.parse_numeric", "parse_numeric", t.name, o.stream_id), n, n,
                    {"unparseable_numeric": _add_flag(flags, bad, "unparseable_numeric")})
        value = self._apply_rules(key, t.name, o.stream_id, o.rules, value, raw.str.strip(), flags)
        if o.scale:
            value = value * o.scale.factor
            j = o.scale.justification
            ss = self._step(f"{key}.scale", "scale", t.name, o.stream_id, function="core.ingest.scale",
                            basis=j.basis, evidence=j.evidence, assumption_ids=j.assumption_ids)
            ss.fixed_params["factor"] = o.scale.factor
            self._count(ss, n, n)
        out_unit = None
        if concept.canonical_unit is not None:
            factor = concept.factor_to_canonical(o.unit)
            if factor != 1.0 or o.unit != concept.canonical_unit:
                value = value * factor
                us = self._step(f"{key}.unit_conversion", "unit_conversion", t.name, o.stream_id,
                                function="core.ingest.unit_conversion")
                us.fixed_params.update({"from_unit": o.unit, "to_unit": concept.canonical_unit, "factor": factor})
                self._count(us, n, n)
            out_unit = concept.canonical_unit
        for code in o.flags:
            self._count(self._step(f"{key}.static_flags", "static_flags", t.name, o.stream_id), n, n,
                        {code: _add_flag(flags, range(n), code)})
        return pd.DataFrame({
            "participant_id": pid.loc[sub_idx].to_numpy(),
            "source_dataset": self.cfg.source,
            "stream_id": o.stream_id,
            "concept": o.concept,
            "timestamp": ts.loc[sub_idx].reset_index(drop=True),
            "value": value.to_numpy(dtype=float),
            "unit": out_unit,
            "raw_value": raw.to_numpy(),
            "is_interpolated": False,
            "quality_flags": flags,
            "source_file": rel,
            "source_row": pf.lines.loc[sub_idx].to_numpy(),
            "source_column": col,
        })

    def _apply_rules(self, key: str, table: str, target: str, rules, numeric: pd.Series | None, text: pd.Series,
                     flags) -> pd.Series | None:
        n = len(text)
        for rule in rules:
            hit = _rule_mask(rule, numeric, text)
            positions = np.flatnonzero(hit)
            raised = _add_flag(flags, positions, rule.flag)
            if self.registry.flags[rule.flag].action is FlagAction.SET_NULL and len(positions) and numeric is not None:
                numeric = numeric.copy()
                numeric.iloc[positions] = np.nan
            self._count(self._rule_step(key, table, target, rule), n, n, {rule.flag: raised})
        return numeric

    # ------------------------------------------------------------- events
    def _events(self, t: TableSpec, e: EventSpec, pf: _ParsedFile, rel, pid, ts, ok) -> pd.DataFrame | None:
        df, hm = pf.df, pf.header_map
        role = f"event:{e.event_type}"
        col = self._resolve(t.name, rel, f"{role}.when_present", e.when_present, hm, required=False)
        if col is None:
            return None
        where = self._where(t, rel, role, df, hm, e.where)
        mask = df[col].notna() & ok & where
        idx = df.index[mask]
        n = len(idx)
        key = f"{t.name}.events.{e.event_type}"
        self._count(self._step(f"{key}.extract", "extract", t.name, e.event_type), int((ok & where).sum()), n,
                    column=col)
        flags = _new_flags(n)
        lines = pf.lines.loc[idx].to_numpy()
        out: dict[str, Any] = {
            "participant_id": pid.loc[idx].to_numpy(),
            "source_dataset": self.cfg.source,
            "event_id": [f"{p}:{e.event_type}:{_file_key(rel)}:{line}" for p, line in zip(pid.loc[idx], lines)],
            "event_type": e.event_type,
            "event_subtype": None,
            "raw_label": None,
            "start_time": ts.loc[idx].reset_index(drop=True),
            "end_time": pd.Series(pd.NaT, index=range(n), dtype=ts.dtype),
            "source_file": rel,
            "source_row": lines,
        }
        if e.subtype:
            scol = self._resolve(t.name, rel, f"{role}.subtype", e.subtype.column, hm, required=False)
            if scol is not None:
                labels = df.loc[idx, scol]
                mapped = labels.str.strip().map(e.subtype.mapping)
                unmapped = np.flatnonzero((labels.notna() & mapped.isna()).to_numpy())
                raised = _add_flag(flags, unmapped, e.subtype.unmapped_flag)
                ms = self._step(f"{key}.map_subtype", "map_category", t.name, e.event_type)
                self._count(ms, int(labels.notna().sum()), int(mapped.notna().sum()),
                            {e.subtype.unmapped_flag: raised}, column=scol)
                out["raw_label"] = labels.where(labels.notna(), None).to_numpy()
                out["event_subtype"] = mapped.where(mapped.notna(), None).to_numpy()
        raw_attrs: dict[str, np.ndarray] = {}
        for at in e.attributes:
            acol = self._resolve(t.name, rel, f"{role}.attribute:{at.target}", at.column, hm, required=False)
            akey = f"{key}.attribute.{at.target}"
            target = f"{e.event_type}.{at.target}"
            if acol is None:
                out[at.target] = np.full(n, np.nan) if at.kind == "numeric" else [None] * n
                raw_attrs[at.target] = np.array([None] * n, dtype=object)
                continue
            raw = df.loc[idx, acol]
            raw_attrs[at.target] = raw.where(raw.notna(), None).to_numpy()
            present = raw.notna().to_numpy()
            if at.kind == "text":
                out[at.target] = raw.where(raw.notna(), None).to_numpy()
                continue
            val = pd.to_numeric(raw.str.strip(), errors="coerce").astype(float)
            bad = np.flatnonzero(present & val.isna().to_numpy())
            self._count(self._step(f"{akey}.parse_numeric", "parse_numeric", t.name, target),
                        int(present.sum()), int(present.sum()),
                        {"unparseable_numeric": _add_flag(flags, bad, "unparseable_numeric")}, column=acol)
            val = self._apply_rules(akey, t.name, target, at.rules, val, raw.str.strip(), flags)
            target_unit = self.event_attr_units.get(at.target)
            if at.unit and target_unit and at.unit != target_unit:
                factor = _units.conversion_factor(at.unit, target_unit)
                val = val * factor
                us = self._step(f"{akey}.unit_conversion", "unit_conversion", t.name, target,
                                function="core.ingest.unit_conversion")
                us.fixed_params.update({"from_unit": at.unit, "to_unit": target_unit, "factor": factor})
                self._count(us, int(present.sum()), int(present.sum()))
            out[at.target] = val.to_numpy(dtype=float)
        out["raw_attributes"] = (
            [json.dumps({k: (None if v[i] is None else str(v[i])) for k, v in raw_attrs.items()}, ensure_ascii=False)
             for i in range(n)] if raw_attrs else [None] * n
        )
        for code in e.flags:
            self._count(self._step(f"{key}.static_flags", "static_flags", t.name, e.event_type), n, n,
                        {code: _add_flag(flags, range(n), code)})
        out["quality_flags"] = flags
        frame = pd.DataFrame(out, index=range(n))
        for c in self.event_columns:  # extension columns not configured for this event type
            if c not in frame.columns:
                frame[c] = np.full(n, np.nan) if c in self.event_float_columns else None
        return frame

    # ------------------------------------------------------------- attributes
    def _attributes(self, t: TableSpec, a: AttributeSpec, pf: _ParsedFile, rel, pid, ok) -> pd.DataFrame | None:
        df, hm = pf.df, pf.header_map
        col = self._resolve(t.name, rel, f"attribute:{a.concept}", a.column, hm, a.required)
        if col is None:
            return None
        concept = self.concepts.get(a.concept)
        idx = df.index[df[col].notna() & ok]
        raw = df.loc[idx, col]
        n = len(idx)
        key = f"{t.name}.attributes.{a.concept}"
        self._count(self._step(f"{key}.extract", "extract", t.name, a.concept), int(ok.sum()), n, column=col)
        flags = _new_flags(n)
        text = raw.str.strip()
        value_num = np.full(n, np.nan)
        value_text: list | np.ndarray = [None] * n
        unit = None
        if concept.value_type is ValueType.CATEGORICAL:
            value_text = text.to_numpy()
            self._apply_rules(key, t.name, a.concept, a.rules, None, text, flags)
        else:
            num = pd.to_numeric(text, errors="coerce").astype(float)
            bad = np.flatnonzero(num.isna().to_numpy())
            self._count(self._step(f"{key}.parse_numeric", "parse_numeric", t.name, a.concept), n, n,
                        {"unparseable_numeric": _add_flag(flags, bad, "unparseable_numeric")})
            num = self._apply_rules(key, t.name, a.concept, a.rules, num, text, flags)
            if concept.canonical_unit is not None:
                factor = concept.factor_to_canonical(a.unit)
                if factor != 1.0 or a.unit != concept.canonical_unit:
                    num = num * factor
                    us = self._step(f"{key}.unit_conversion", "unit_conversion", t.name, a.concept,
                                    function="core.ingest.unit_conversion")
                    us.fixed_params.update({"from_unit": a.unit, "to_unit": concept.canonical_unit, "factor": factor})
                    self._count(us, n, n)
                unit = concept.canonical_unit
            value_num = num.to_numpy(dtype=float)
        for code in a.flags:
            self._count(self._step(f"{key}.static_flags", "static_flags", t.name, a.concept), n, n,
                        {code: _add_flag(flags, range(n), code)})
        measured_raw = [None] * n
        if a.measured_at_raw:
            mcol = self._resolve(t.name, rel, f"attribute:{a.concept}.measured_at_raw", a.measured_at_raw, hm, False)
            if mcol is not None:
                m = df.loc[idx, mcol]
                measured_raw = m.where(m.notna(), None).to_numpy()
        return pd.DataFrame({
            "participant_id": pid.loc[idx].to_numpy(),
            "source_dataset": self.cfg.source,
            "concept": a.concept,
            "value_num": value_num,
            "value_text": value_text,
            "unit": unit,
            "raw_value": raw.to_numpy(),
            "measured_at": pd.Series(pd.NaT, index=range(n), dtype=_ts_dtype(self.tz)),
            "measured_at_raw": measured_raw,
            "quality_flags": flags,
            "source_file": rel,
            "source_row": pf.lines.loc[idx].to_numpy(),
            "source_column": col,
        })

    # ------------------------------------------------------------- participants
    def _participants(self, t: TableSpec, rel: str, df: pd.DataFrame, hm) -> tuple[pd.Series, pd.Series]:
        spec = t.participant_id
        if spec.from_ == "path":
            m = re.search(spec.regex, rel)
            if not m:
                raise IngestError(f"{rel}: participant regex {spec.regex!r} does not match the path.")
            pid = f"{self.cfg.source}:{spec.format.format(m.group(1))}"
            return pd.Series(pid, index=df.index, dtype=object), pd.Series(True, index=df.index)
        col = self._resolve(t.name, rel, "participant_id", spec.column, hm, required=True)
        local = df[col].str.strip()
        ok = local.notna() & (local != "")
        pid = pd.Series(None, index=df.index, dtype=object)
        pid[ok] = local[ok].map(lambda v: f"{self.cfg.source}:{spec.format.format(v)}")
        return pid, ok

    # ------------------------------------------------------------- one file
    def _process_file(self, t: TableSpec, path: Path, parts: dict) -> None:
        rel = path.relative_to(self.data_root).as_posix()
        self.input_files.append(path)
        pf = self._read(t, path, rel)
        self._used_cols = set()
        for line, fields in pf.overlong:
            self._reject(t, rel, line, "field_count_mismatch", json.dumps({"fields": fields}, ensure_ascii=False),
                         detail=f"{len(fields)} fields but the header has {len(pf.df.columns)} columns")
        original = pf.df
        dropped = []
        for alias in t.read.drop_columns:
            c = self._resolve(t.name, rel, f"drop:{alias}", (alias,), pf.header_map, required=False)
            if c is not None:
                dropped.append(c)
        if dropped:
            pf.df = pf.df.drop(columns=dropped)
            pf.header_map = {k: v for k, v in pf.header_map.items() if v not in dropped}

        rs = self._step(f"{t.name}.read", "read", t.name, None, function="core.ingest.read_delimited")
        rs.notes.add("all values read as text; source_row = physical start line (header = 1)")
        self._count(rs, pf.n_records, len(pf.df), encoding=t.read.encoding, delimiter=t.read.delimiter)
        rs.fixed_params["files"] = rs.fixed_params.get("files", 0) + 1
        rs.fixed_params["blank_lines_skipped"] = rs.fixed_params.get("blank_lines_skipped", 0) + pf.n_blank_lines
        rs.fixed_params["short_rows_padded"] = rs.fixed_params.get("short_rows_padded", 0) + pf.n_padded

        df = pf.df
        empty = df.isna().all(axis=1) if len(df.columns) else pd.Series(True, index=df.index)
        pid, pid_ok = self._participants(t, rel, df, pf.header_map)
        reason = pd.Series(None, index=df.index, dtype=object)
        reason[empty] = "empty_row"
        reason[reason.isna() & ~pid_ok] = "missing_participant_id"
        ps = self._step(f"{t.name}.participant_id", "participant_id", t.name, None)
        self._count(ps, int((~empty).sum()), int((~empty & pid_ok).sum()))
        ts = None
        if t.timestamp:
            tcol = self._resolve(t.name, rel, "timestamp", t.timestamp.column, pf.header_map, required=True)
            ts, ts_reason = self._parse_timestamps(t, rel, df[tcol])
            fill = reason.isna() & ts_reason.notna()
            reason[fill] = ts_reason[fill]
        ok = reason.isna()
        for i in df.index[~ok]:
            self._reject(t, rel, pf.lines[i], reason[i], _json_row(original.loc[i]),
                         participant_id=pid[i] if pid_ok[i] else None)

        for o in t.observations:
            part = self._observations(t, o, pf, rel, pid, ts, ok)
            if part is not None and len(part):
                parts["obs"].setdefault(o.stream_id, []).append(part)
        for e in t.events:
            part = self._events(t, e, pf, rel, pid, ts, ok)
            if part is not None and len(part):
                parts["events"].append(part)
        for a in t.attributes:
            part = self._attributes(t, a, pf, rel, pid, ok)
            if part is not None and len(part):
                parts["attrs"].append(part)

        # values of row-filter columns that no observation/event spec selects (e.g. a label with different case)
        selected: dict[str, set[str]] = {}
        for spec in list(t.observations) + list(t.events):
            if spec.where is not None:
                wcol = next((pf.header_map[normalize_header(a)] for a in spec.where.column
                             if normalize_header(a) in pf.header_map), None)
                if wcol is not None:
                    selected.setdefault(wcol, set()).update(spec.where.values)
        for wcol, vals in selected.items():
            counts = df.loc[ok, wcol].str.strip().fillna("<empty>").value_counts()
            for v, cnt in counts.items():
                if v not in vals:
                    self.unselected.append({"table": t.name, "source_file": rel, "column": wcol, "value": v,
                                            "n_rows": int(cnt)})

        used = self._used_cols
        self.file_stats.append({
            "table": t.name, "source_file": rel, "n_physical_lines": pf.n_physical_lines, "n_records": pf.n_records,
            "n_blank_lines": pf.n_blank_lines, "n_short_rows_padded": pf.n_padded,
            "n_rejected": int((~ok).sum()) + len(pf.overlong), "n_participants": int(pid[ok].nunique()),
            "dropped_columns": "|".join(dropped),
            "unused_columns": "|".join(c for c in df.columns if c not in used),
        })

    # ------------------------------------------------------------- transforms
    def _run_transforms(self, t: TableSpec, o: ObservationSpec, frame: pd.DataFrame) -> pd.DataFrame:
        for tr in o.transforms:
            fn = get_transform(tr.name)
            outs, rows_in, rows_out, delta, n_removed = [], 0, 0, Counter(), 0
            for pid, group in frame.groupby("participant_id", sort=True):
                g = group.reset_index(drop=True)
                ctx = TransformContext(self.cfg.source, pid, o.stream_id, o.concept)
                res = fn(g.copy(), dict(tr.params), ctx)
                removed, added = self._check_transform_output(tr, g, res)
                for _, r in removed.iterrows():
                    self._reject(t, r["source_file"], r["source_row"], "removed_by_transform", json.dumps({
                        "stream_id": r["stream_id"], "timestamp": str(r["timestamp"]), "raw_value": r["raw_value"],
                        "value": None if pd.isna(r["value"]) else float(r["value"]),
                        "source_column": r["source_column"], "quality_flags": list(r["quality_flags"]),
                    }, ensure_ascii=False), participant_id=pid, detail=f"transform {tr.name}", scope="observation")
                delta.update(added)
                rows_in += len(g)
                rows_out += len(res)
                n_removed += len(removed)
                outs.append(res)
            frame = pd.concat(outs, ignore_index=True) if outs else frame
            j = tr.justification
            params = dict(tr.params)
            params["declared_flags"] = list(tr.flags)
            self.log.record(f"{t.name}.observations.{o.stream_id}.transform.{tr.name}", tr.name, j.basis, rows_in,
                            rows_out, params=params, assumption_ids=list(j.assumption_ids), evidence=j.evidence,
                            flags_raised=dict(delta), operation="transform", table=t.name, target=o.stream_id,
                            notes=f"{n_removed} observation(s) removed; kept in rejected_records" if n_removed else None)
        return frame

    def _check_transform_output(self, tr: TransformSpec, before: pd.DataFrame, after: pd.DataFrame):
        """Enforce the transform contract; return (removed rows, Counter of flags added)."""
        name = tr.name
        if list(after.columns) != list(before.columns):
            raise IngestError(f"Transform {name} changed the table columns.")
        keys = ["source_file", "source_row", "timestamp"]
        if after.duplicated(keys).any():
            raise IngestError(f"Transform {name} produced duplicate (source_file, source_row, timestamp) keys.")
        declared, explanatory = set(tr.flags), self._explanatory_flags(tr)
        m = after.reset_index(drop=True).merge(before, on=keys, how="left", suffixes=("", "_before"), indicator=True)
        is_new = (m["_merge"] == "left_only").to_numpy()
        added_counter: Counter = Counter()
        explained_list = []
        for i, (flags, prev) in enumerate(zip(m["quality_flags"], m["quality_flags_before"])):
            added = set(flags) - (set() if is_new[i] else set(prev))
            undeclared = added - declared
            if undeclared:
                raise IngestError(f"Transform {name} raised flag(s) {sorted(undeclared)} that are not declared in "
                                  "its 'flags'.")
            added_counter.update(added)
            if is_new[i] and not (set(flags) & explanatory):
                raise IngestError(f"Transform {name} created a row without an explanatory flag "
                                  f"(allowed: {sorted(explanatory) or 'none'}).")
            explained_list.append(bool(added & explanatory))
        m["_explained"] = explained_list
        old = m[~is_new]
        if len(old):
            for c in _PROTECTED:
                if not _same(old[c], old[f"{c}_before"]).all():
                    raise IngestError(f"Transform {name} modified protected column {c!r}.")
            changed = ~(_same(old["value"], old["value_before"]) & _same(old["is_interpolated"], old["is_interpolated_before"]))
            explained = old["_explained"]
            if (changed & ~explained).any():
                raise IngestError(
                    f"Transform {name} changed values without adding an explanatory flag declared for it "
                    f"(explanatory flags: {sorted(explanatory) or 'none'}; basis {tr.justification.basis.value!r}).")
        back = before.merge(after[keys], on=keys, how="left", indicator=True)
        removed = before[(back["_merge"] == "left_only").to_numpy()]
        return removed, added_counter

    # ------------------------------------------------------------- driver
    def run(self) -> IngestResult:
        parts: dict[str, Any] = {"obs": {}, "events": [], "attrs": []}
        for t in self.cfg.tables:
            for path in self._files(t):
                self._process_file(t, path, parts)
            self._flush_steps()

        observations, stream_rows = self._finalise_observations(parts["obs"])
        events = pd.concat(parts["events"], ignore_index=True) if parts["events"] else _empty(self.event_columns)
        attrs = pd.concat(parts["attrs"], ignore_index=True) if parts["attrs"] else _empty(ATTR_COLUMNS)
        streams = pd.DataFrame(stream_rows, columns=STREAM_COLUMNS) if stream_rows else _empty(STREAM_COLUMNS)
        rejected = _typed(pd.DataFrame(self.rejected, columns=REJECT_COLUMNS) if self.rejected
                          else _empty(REJECT_COLUMNS), REJECT_COLUMNS, self.tz, [], float_cols=())
        result = IngestResult(
            config=self.cfg, concepts=self.concepts, registry=self.registry,
            observations=_typed(observations, OBS_COLUMNS, self.tz, ["timestamp"]),
            events=_typed(events[self.event_columns], self.event_columns, self.tz, ["start_time", "end_time"],
                          float_cols=self.event_float_columns),
            participant_attributes=_typed(attrs, ATTR_COLUMNS, self.tz, ["measured_at"]),
            streams=_typed(streams, STREAM_COLUMNS, self.tz, ["first_timestamp", "last_timestamp"]),
            rejected_records=rejected,
            file_stats=pd.DataFrame(self.file_stats, columns=FILE_STAT_COLUMNS),
            column_resolution=self.resolution_frame(),
            unselected_values=self.unselected_frame(),
            transform_log=self.log, input_files=self.input_files, data_root=self.data_root,
            event_schema_fn=self.event_schema_fn,
        )
        validate_result(result)
        return result

    def resolution_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.resolution, columns=["table", "file", "role", "aliases", "matched_column", "status"])

    def unselected_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.unselected, columns=["table", "source_file", "column", "value", "n_rows"])

    def _finalise_observations(self, parts: dict[str, list[pd.DataFrame]]):
        frames, stream_rows = [], []
        for t in self.cfg.tables:
            for o in t.observations:
                if not parts.get(o.stream_id):
                    continue
                frame = pd.concat(parts[o.stream_id], ignore_index=True)
                frame["source_row"] = frame["source_row"].astype("Int64")
                source_counts = frame.groupby("participant_id").size()
                frame = self._run_transforms(t, o, frame)
                for pid, g in frame.groupby("participant_id", sort=True):
                    stream_rows.append({
                        "participant_id": pid, "source_dataset": self.cfg.source, "stream_id": o.stream_id,
                        "concept": o.concept, "device": o.device,
                        "declared_native_interval_s": None if o.native_interval_s is None else float(o.native_interval_s),
                        "first_timestamp": g["timestamp"].min(), "last_timestamp": g["timestamp"].max(),
                        "n_rows_source": int(source_counts.get(pid, 0)), "n_rows_output": len(g),
                        "quality_flags": [],
                    })
                frames.append(frame)
        obs = pd.concat(frames, ignore_index=True) if frames else _empty(OBS_COLUMNS)
        return obs, stream_rows


def _ts_dtype(tz: str | None):
    return "datetime64[ns]" if tz is None else pd.DatetimeTZDtype("ns", tz)


def _as_ts(s: pd.Series, tz: str | None) -> pd.Series:
    """Bring a timestamp column to the output temporal policy (type change only, never a guess)."""
    if len(s) == 0 or s.isna().all():
        return pd.Series(pd.NaT, index=s.index, dtype=_ts_dtype(tz))
    if not pd.api.types.is_datetime64_any_dtype(s):
        s = pd.to_datetime(s)
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        if tz is None:
            raise IngestError("timezone-aware timestamps in a naive-time output")
        return s.dt.tz_convert(tz).dt.as_unit("ns")
    if tz is not None:
        raise IngestError("naive timestamps in a timezone-aware output")
    return s.dt.as_unit("ns")


def _typed(df: pd.DataFrame, columns: list[str], tz: str | None, ts_cols: list[str],
           float_cols: tuple[str, ...] = ("value", "value_num", "declared_native_interval_s")) -> pd.DataFrame:
    df = df.reindex(columns=columns).reset_index(drop=True)
    for c in ts_cols:
        if c in df:
            df[c] = _as_ts(df[c], tz)
    for c in float_cols:
        if c in df:
            df[c] = pd.to_numeric(df[c]).astype(float)
    for c in ("source_row", "n_rows_source", "n_rows_output"):
        if c in df:
            df[c] = df[c].astype("Int64")
    if "is_interpolated" in df:
        df["is_interpolated"] = df["is_interpolated"].astype(bool)
    for c in columns:
        if df[c].dtype == object and c != "quality_flags":
            df[c] = df[c].where(df[c].notna(), None)
    if "quality_flags" in df and len(df) == 0:
        df["quality_flags"] = pd.Series(dtype=object)
    return df


# =========================================================================== public API
def validate_result(result: IngestResult) -> None:
    """Schema, concept and flag validation of all output tables (raises on problems)."""
    tz = result.config.temporal.output_timezone
    tables = {
        "observations": (result.observations, observation_schema(tz), ConceptKind.OBSERVATION),
        "events": (result.events, result.event_schema_fn(tz), None),
        "participant_attributes": (result.participant_attributes, participant_attribute_schema(tz), ConceptKind.ATTRIBUTE),
        "streams": (result.streams, stream_schema(tz), None),
    }
    for name, (df, schema, kind) in tables.items():
        schema.validate(df, lazy=True)
        validate_flags(df, result.registry)
        if kind is not None and len(df):
            problems = check_table_concepts(df, result.concepts, kind)
            if problems:
                raise IngestError(f"{name}: concept problems: {problems}")
    rejected_record_schema(tz).validate(result.rejected_records, lazy=True)


def ingest_source(config: SourceConfig, data_root: str | Path, concepts: ConceptRegistry,
                  clock=_utc_now) -> IngestResult:
    """Ingest a source described by ``config`` from files under ``data_root``."""
    return _Engine(config, Path(data_root), concepts, clock).run()


OUTPUT_FILES = (
    "observations.parquet", "events.parquet", "participant_attributes.parquet", "streams.parquet",
    "rejected_records.parquet", "ingest_summary.json", "ingestion_report.md",
    "_provenance/manifest.json", "_provenance/transform_log.jsonl", "_provenance/column_resolution.csv",
    "_provenance/file_stats.csv", "_provenance/unselected_filter_values.csv",
)


def write_outputs(result: IngestResult, out_dir: str | Path, base_dir: str | Path, clock=_utc_now,
                  record_git: bool = True, config_path: str | Path | None = None) -> dict[str, Any]:
    """Write validated tables, audit files and the Markdown report; returns the manifest as a dict.

    Layout::

        out_dir/{observations,events,participant_attributes,streams,rejected_records}.parquet
        out_dir/ingestion_report.md                (human-readable run report)
        out_dir/ingest_summary.json                (aggregate only, machine-readable)
        out_dir/_provenance/manifest.json
        out_dir/_provenance/transform_log.jsonl
        out_dir/_provenance/column_resolution.csv
        out_dir/_provenance/file_stats.csv
    """
    from .report import ReportContext, render_report  # local import avoids a cycle

    out_dir = Path(out_dir)
    prov = out_dir / "_provenance"
    prov.mkdir(parents=True, exist_ok=True)
    tz = result.config.temporal.output_timezone
    manifest = RunManifest(f"ingest:{result.config.source}", base_dir, clock=clock, record_git=record_git)
    manifest.add_config("source_config", result.config.raw)
    manifest.add_config("concepts", {cid: repr(c) for cid, c in sorted(result.concepts.concepts.items())})
    base = Path(base_dir).resolve()

    def _display(p: Path) -> str:  # relative to base_dir when possible; never an absolute path
        try:
            return p.resolve().relative_to(base).as_posix()
        except ValueError:
            return p.resolve().name

    manifest.set_source({
        "source": result.config.source, "config_version": result.config.version,
        "data_root": _display(result.data_root), "output_dir": _display(out_dir),
        "input_paths_relative_to": "data_root", "output_paths_relative_to": "output_dir",
        "output_timezone": tz, "date_shifted": result.config.temporal.date_shifted,
        "temporal_notes": result.config.temporal.notes,
    })
    manifest.set_annotations(result.registry)
    for p in result.input_files:
        manifest.add_input(p, relative_to=result.data_root)
    tables = {
        "observations": (result.observations, observation_schema(tz)),
        "events": (result.events, result.event_schema_fn(tz)),
        "participant_attributes": (result.participant_attributes, participant_attribute_schema(tz)),
        "streams": (result.streams, stream_schema(tz)),
        "rejected_records": (result.rejected_records, rejected_record_schema(tz)),
    }
    for name, (df, schema) in tables.items():
        path = out_dir / f"{name}.parquet"
        write_table(df, schema, path)
        manifest.add_output(path, table=name, rows=len(df), relative_to=out_dir)
    result.transform_log.write_jsonl(prov / "transform_log.jsonl")
    result.column_resolution.to_csv(prov / "column_resolution.csv", index=False)
    result.file_stats.to_csv(prov / "file_stats.csv", index=False)
    result.unselected_values.to_csv(prov / "unselected_filter_values.csv", index=False)
    (out_dir / "ingest_summary.json").write_text(canonical_json(result.summary()) + "\n", encoding="utf-8")
    report = render_report(ReportContext.from_result(result, out_dir=out_dir, base_dir=base,
                                                     manifest=manifest.data, config_path=config_path))
    (out_dir / "ingestion_report.md").write_text(report, encoding="utf-8")
    for extra in ("_provenance/transform_log.jsonl", "_provenance/column_resolution.csv", "_provenance/file_stats.csv",
                  "_provenance/unselected_filter_values.csv", "ingest_summary.json", "ingestion_report.md"):
        manifest.add_output(out_dir / extra, relative_to=out_dir)
    manifest.write(prov / "manifest.json")
    return manifest.to_dict()


@dataclass
class IngestOutcome:
    status: str  # "success" | "warnings" | "failed"
    result: IngestResult | None
    report_path: Path
    error: str | None = None


def ingest_and_write(config: SourceConfig | str | Path, concepts: ConceptRegistry | str | Path,
                     data_root: str | Path, out_dir: str | Path, base_dir: str | Path, clock=_utc_now,
                     record_git: bool = True) -> IngestOutcome:
    """Run ingestion end to end and always leave an ``ingestion_report.md`` in ``out_dir``.

    Outputs of any previous run in ``out_dir`` (the files in :data:`OUTPUT_FILES`) are removed first, so the
    directory never mixes results from different runs. On failure only the report is written and the
    exception is re-raised.
    """
    from .concepts import ConceptRegistry as _CR
    from .report import ReportContext, determine_status, render_report

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name in OUTPUT_FILES:
        (out_dir / name).unlink(missing_ok=True)
    config_path = None if isinstance(config, SourceConfig) else Path(config)
    engine = None
    try:
        cfg = config if isinstance(config, SourceConfig) else SourceConfig.from_yaml(config)
        reg = concepts if isinstance(concepts, _CR) else _CR.from_yaml(concepts)
        engine = _Engine(cfg, Path(data_root), reg, clock)
        result = engine.run()
        write_outputs(result, out_dir, base_dir, clock=clock, record_git=record_git, config_path=config_path)
        status, _ = determine_status(ReportContext.from_result(result, out_dir=out_dir, base_dir=Path(base_dir)))
        return IngestOutcome(status, result, out_dir / "ingestion_report.md")
    except Exception as exc:
        ctx = ReportContext.from_failure(exc, engine=engine, config=config, out_dir=out_dir,
                                         base_dir=Path(base_dir), data_root=Path(data_root), config_path=config_path)
        (out_dir / "ingestion_report.md").write_text(render_report(ctx), encoding="utf-8")
        raise
