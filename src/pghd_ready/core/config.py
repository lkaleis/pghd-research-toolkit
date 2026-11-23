"""Source configuration: parsing and structural validation of source YAML.

A source configuration declares *what* to extract from a set of tabular files;
:mod:`pghd_ready.core.ingest` implements *how*. Unknown keys are rejected so
that typos never silently change behaviour.

Column references are alias lists: the first alias present in a file is used.
Header matching is case-insensitive and ignores surrounding/duplicate
whitespace (see :func:`normalize_header`).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import yaml

from .annotations import AnnotationRegistry
from .provenance import TransformBasis


class ConfigError(ValueError):
    pass


def normalize_header(name: str) -> str:
    return re.sub(r"\s+", " ", str(name).replace("﻿", "")).strip().casefold()


def _check_keys(where: str, spec: Mapping[str, Any], allowed: set[str], required: set[str] = frozenset()) -> None:
    if not isinstance(spec, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {type(spec).__name__}.")
    unknown = set(spec) - allowed
    if unknown:
        raise ConfigError(f"{where}: unknown keys {sorted(unknown)} (allowed: {sorted(allowed)}).")
    missing = set(required) - set(spec)
    if missing:
        raise ConfigError(f"{where}: missing required keys {sorted(missing)}.")


def _aliases(where: str, value: Any) -> tuple[str, ...]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v.strip() for v in value):
        raise ConfigError(f"{where}: column must be a non-empty string or list of strings.")
    return tuple(value)


# --------------------------------------------------------------------------- small specs
@dataclass(frozen=True)
class BasisSpec:
    """Justification for a value-changing step (scale, transform)."""

    basis: TransformBasis
    evidence: str | None = None
    assumption_ids: tuple[str, ...] = ()

    @classmethod
    def parse(cls, where: str, spec: Mapping[str, Any]) -> "BasisSpec":
        try:
            basis = TransformBasis(spec.get("basis"))
        except ValueError as exc:
            raise ConfigError(f"{where}: basis must be one of {[b.value for b in TransformBasis]}.") from exc
        ids = tuple(spec.get("assumption_ids") or ())
        evidence = spec.get("evidence")
        if basis is TransformBasis.DOCUMENTED and not evidence:
            raise ConfigError(f"{where}: documented basis requires 'evidence'.")
        if basis is TransformBasis.ASSUMPTION and not ids:
            raise ConfigError(f"{where}: assumption basis requires 'assumption_ids'.")
        if basis is not TransformBasis.ASSUMPTION and ids:
            raise ConfigError(f"{where}: 'assumption_ids' only allowed with basis 'assumption'.")
        return cls(basis, evidence, ids)


@dataclass(frozen=True)
class ScaleSpec:
    factor: float
    justification: BasisSpec

    @classmethod
    def parse(cls, where: str, spec: Mapping[str, Any]) -> "ScaleSpec":
        _check_keys(where, spec, {"factor", "basis", "evidence", "assumption_ids"}, {"factor", "basis"})
        factor = spec["factor"]
        if not isinstance(factor, (int, float)) or factor == 0:
            raise ConfigError(f"{where}: factor must be a non-zero number.")
        just = BasisSpec.parse(where, spec)
        if just.basis is TransformBasis.STRUCTURAL:
            raise ConfigError(f"{where}: rescaling changes values; basis must be 'documented' or 'assumption'.")
        return cls(float(factor), just)


_RULE_OPS = {"equals", "in", "below", "above", "outside"}


@dataclass(frozen=True)
class SamplingOriginSpec:
    """Declared sampling origin of every reading in a stream.

    Without a declaration, ``is_interpolated`` is *unknown* (null). A declaration sets it for the whole
    stream and must be justified (documented with evidence, or assumption with assumption_ids).
    Per-reading classification (e.g. inferring interpolated readings) is the job of a named transform.
    """

    value: str  # "native" | "interpolated"
    justification: BasisSpec

    @classmethod
    def parse(cls, where: str, spec: Mapping[str, Any]) -> "SamplingOriginSpec":
        _check_keys(where, spec, {"value", "basis", "evidence", "assumption_ids"}, {"value", "basis"})
        if spec["value"] not in ("native", "interpolated"):
            raise ConfigError(f"{where}: value must be 'native' or 'interpolated' (omit the key for unknown).")
        just = BasisSpec.parse(where, spec)
        if just.basis is TransformBasis.STRUCTURAL:
            raise ConfigError(f"{where}: a sampling-origin declaration needs basis 'documented' or 'assumption'.")
        return cls(spec["value"], just)


def _domains(where: str, value: Any) -> tuple[str, ...]:
    from .concepts import ConceptError, validate_domains

    try:
        return validate_domains(where, value)
    except ConceptError as exc:
        raise ConfigError(str(exc)) from exc


@dataclass(frozen=True)
class RuleSpec:
    """Flag values meeting a condition. Evaluated on the parsed *source* value (before scaling/units)."""

    op: str
    operand: Any
    flag: str

    @classmethod
    def parse(cls, where: str, spec: Mapping[str, Any]) -> "RuleSpec":
        _check_keys(where, spec, {"when", "flag"}, {"when", "flag"})
        when = spec["when"]
        if not isinstance(when, Mapping) or len(when) != 1 or next(iter(when)) not in _RULE_OPS:
            raise ConfigError(f"{where}: 'when' must have exactly one of {sorted(_RULE_OPS)}.")
        op, operand = next(iter(when.items()))
        if op == "in" and not isinstance(operand, list):
            raise ConfigError(f"{where}: 'in' needs a list.")
        if op == "outside" and not (isinstance(operand, list) and len(operand) == 2):
            raise ConfigError(f"{where}: 'outside' needs [low, high].")
        return cls(op, operand, spec["flag"])


@dataclass(frozen=True)
class WhereSpec:
    """Row filter, e.g. for long-format sources: {column: [type], equals: heart_rate}."""

    column: tuple[str, ...]
    values: tuple[str, ...]

    @classmethod
    def parse(cls, where: str, spec: Mapping[str, Any]) -> "WhereSpec":
        _check_keys(where, spec, {"column", "equals", "in"}, {"column"})
        if ("equals" in spec) == ("in" in spec):
            raise ConfigError(f"{where}: give exactly one of 'equals' or 'in'.")
        vals = [spec["equals"]] if "equals" in spec else list(spec["in"])
        return cls(_aliases(f"{where}.column", spec["column"]), tuple(str(v) for v in vals))


@dataclass(frozen=True)
class TransformSpec:
    """A named transform. ``flags`` lists every flag code the transform is allowed to raise.

    A transform may change a value (or ``is_interpolated``) only on rows to which it adds one of
    its declared *explanatory* flags: an ``assumption_based`` flag whose assumption is listed in
    ``assumption_ids``, or a ``documented_issue`` flag when the basis is ``documented``.
    """

    name: str
    params: Mapping[str, Any]
    justification: BasisSpec
    flags: tuple[str, ...] = ()

    @classmethod
    def parse(cls, where: str, spec: Mapping[str, Any]) -> "TransformSpec":
        _check_keys(where, spec, {"name", "params", "basis", "evidence", "assumption_ids", "flags"}, {"name", "basis"})
        return cls(spec["name"], dict(spec.get("params") or {}), BasisSpec.parse(where, spec),
                   tuple(spec.get("flags") or ()))


# --------------------------------------------------------------------------- record specs
@dataclass(frozen=True)
class ObservationSpec:
    stream_id: str
    concept: str
    column: tuple[str, ...]
    unit: str | None = None
    required: bool = False
    device: str | None = None
    native_interval_s: float | None = None
    where: WhereSpec | None = None
    scale: ScaleSpec | None = None
    rules: tuple[RuleSpec, ...] = ()
    flags: tuple[str, ...] = ()
    transforms: tuple[TransformSpec, ...] = ()
    sampling_origin: SamplingOriginSpec | None = None

    @classmethod
    def parse(cls, where: str, spec: Mapping[str, Any]) -> "ObservationSpec":
        _check_keys(
            where, spec,
            {"stream_id", "concept", "column", "unit", "required", "device", "native_interval_s", "where",
             "scale", "rules", "flags", "transforms", "sampling_origin"},
            {"stream_id", "concept", "column"},
        )
        w = f"{where}[{spec['stream_id']}]"
        return cls(
            stream_id=spec["stream_id"],
            concept=spec["concept"],
            column=_aliases(f"{w}.column", spec["column"]),
            unit=spec.get("unit"),
            required=bool(spec.get("required", False)),
            device=spec.get("device"),
            native_interval_s=spec.get("native_interval_s"),
            where=WhereSpec.parse(f"{w}.where", spec["where"]) if spec.get("where") else None,
            scale=ScaleSpec.parse(f"{w}.scale", spec["scale"]) if spec.get("scale") else None,
            rules=tuple(RuleSpec.parse(f"{w}.rules[{i}]", r) for i, r in enumerate(spec.get("rules") or [])),
            flags=tuple(spec.get("flags") or ()),
            transforms=tuple(TransformSpec.parse(f"{w}.transforms[{i}]", t)
                             for i, t in enumerate(spec.get("transforms") or [])),
            sampling_origin=SamplingOriginSpec.parse(f"{w}.sampling_origin", spec["sampling_origin"])
            if spec.get("sampling_origin") else None,
        )


@dataclass(frozen=True)
class EventAttributeSpec:
    target: str
    column: tuple[str, ...]
    kind: str = "numeric"  # numeric | text
    unit: str | None = None
    rules: tuple[RuleSpec, ...] = ()

    @classmethod
    def parse(cls, where: str, target: str, spec: Any) -> "EventAttributeSpec":
        if isinstance(spec, (str, list)):
            spec = {"column": spec}
        _check_keys(where, spec, {"column", "kind", "unit", "rules"}, {"column"})
        kind = spec.get("kind", "numeric")
        if kind not in ("numeric", "text"):
            raise ConfigError(f"{where}: kind must be 'numeric' or 'text'.")
        if kind == "text" and (spec.get("unit") or spec.get("rules")):
            raise ConfigError(f"{where}: text attributes take no unit or rules.")
        return cls(target, _aliases(f"{where}.column", spec["column"]), kind, spec.get("unit"),
                   tuple(RuleSpec.parse(f"{where}.rules[{i}]", r) for i, r in enumerate(spec.get("rules") or [])))


@dataclass(frozen=True)
class CategoryMapSpec:
    column: tuple[str, ...]
    mapping: Mapping[str, str]  # raw label -> canonical
    unmapped_flag: str = "unmapped_category"

    @classmethod
    def parse(cls, where: str, spec: Mapping[str, Any]) -> "CategoryMapSpec":
        _check_keys(where, spec, {"column", "map", "unmapped_flag"}, {"column", "map"})
        lookup: dict[str, str] = {}
        for canonical, raws in spec["map"].items():
            for raw in ([raws] if isinstance(raws, str) else raws):
                if raw in lookup and lookup[raw] != canonical:
                    raise ConfigError(f"{where}: raw label {raw!r} mapped to two categories.")
                lookup[str(raw)] = canonical
        return cls(_aliases(f"{where}.column", spec["column"]), lookup, spec.get("unmapped_flag", "unmapped_category"))


@dataclass(frozen=True)
class EventSpec:
    event_type: str
    when_present: tuple[str, ...]
    where: WhereSpec | None = None
    subtype: CategoryMapSpec | None = None
    attributes: tuple[EventAttributeSpec, ...] = ()
    flags: tuple[str, ...] = ()
    domains: tuple[str, ...] = ()  # descriptive PGHD domain tags (many-to-many)

    @classmethod
    def parse(cls, where: str, spec: Mapping[str, Any]) -> "EventSpec":
        _check_keys(where, spec, {"event_type", "when_present", "where", "subtype", "attributes", "flags", "domains"},
                    {"event_type", "when_present"})
        w = f"{where}[{spec['event_type']}]"
        return cls(
            event_type=spec["event_type"],
            when_present=_aliases(f"{w}.when_present", spec["when_present"]),
            where=WhereSpec.parse(f"{w}.where", spec["where"]) if spec.get("where") else None,
            subtype=CategoryMapSpec.parse(f"{w}.subtype", spec["subtype"]) if spec.get("subtype") else None,
            attributes=tuple(EventAttributeSpec.parse(f"{w}.attributes.{k}", k, v)
                             for k, v in (spec.get("attributes") or {}).items()),
            flags=tuple(spec.get("flags") or ()),
            domains=_domains(f"{w}.domains", spec.get("domains")),
        )


@dataclass(frozen=True)
class AttributeSpec:
    concept: str
    column: tuple[str, ...]
    unit: str | None = None
    required: bool = False
    rules: tuple[RuleSpec, ...] = ()
    flags: tuple[str, ...] = ()
    measured_at_raw: tuple[str, ...] | None = None

    @classmethod
    def parse(cls, where: str, spec: Mapping[str, Any]) -> "AttributeSpec":
        _check_keys(where, spec, {"concept", "column", "unit", "required", "rules", "flags", "measured_at_raw"},
                    {"concept", "column"})
        w = f"{where}[{spec['concept']}]"
        return cls(
            concept=spec["concept"],
            column=_aliases(f"{w}.column", spec["column"]),
            unit=spec.get("unit"),
            required=bool(spec.get("required", False)),
            rules=tuple(RuleSpec.parse(f"{w}.rules[{i}]", r) for i, r in enumerate(spec.get("rules") or [])),
            flags=tuple(spec.get("flags") or ()),
            measured_at_raw=_aliases(f"{w}.measured_at_raw", spec["measured_at_raw"])
            if spec.get("measured_at_raw") else None,
        )


# --------------------------------------------------------------------------- table / source
@dataclass(frozen=True)
class ParticipantIdSpec:
    from_: str  # "path" | "column"
    regex: str | None = None
    column: tuple[str, ...] | None = None
    format: str = "{0}"

    @classmethod
    def parse(cls, where: str, spec: Mapping[str, Any]) -> "ParticipantIdSpec":
        _check_keys(where, spec, {"from", "regex", "column", "format"}, {"from"})
        src = spec["from"]
        if src == "path":
            if not spec.get("regex"):
                raise ConfigError(f"{where}: from: path requires 'regex' with one capture group.")
            if re.compile(spec["regex"]).groups != 1:
                raise ConfigError(f"{where}: regex must have exactly one capture group.")
            return cls("path", regex=spec["regex"], format=spec.get("format", "{0}"))
        if src == "column":
            return cls("column", column=_aliases(f"{where}.column", spec.get("column")), format=spec.get("format", "{0}"))
        raise ConfigError(f"{where}: 'from' must be 'path' or 'column'.")


@dataclass(frozen=True)
class ReadSpec:
    encoding: str = "utf-8-sig"
    delimiter: str = ","
    na_values: tuple[str, ...] = ("",)
    drop_columns: tuple[str, ...] = ()

    @classmethod
    def parse(cls, where: str, spec: Mapping[str, Any] | None) -> "ReadSpec":
        spec = spec or {}
        _check_keys(where, spec, {"encoding", "delimiter", "na_values", "drop_columns"})
        return cls(
            encoding=spec.get("encoding", "utf-8-sig"),
            delimiter=spec.get("delimiter", ","),
            na_values=tuple(str(v) for v in spec.get("na_values", [""])),
            drop_columns=tuple(spec.get("drop_columns") or ()),
        )


@dataclass(frozen=True)
class TimestampSpec:
    column: tuple[str, ...]
    format: str  # strftime format or "ISO8601"
    assume_timezone: str | None = None  # localise naive source times to this zone (explicit, never inferred)
    timezone_justification: BasisSpec | None = None  # required with assume_timezone

    @classmethod
    def parse(cls, where: str, spec: Mapping[str, Any]) -> "TimestampSpec":
        _check_keys(where, spec, {"column", "format", "assume_timezone", "basis", "evidence", "assumption_ids"},
                    {"column", "format"})
        just = None
        if spec.get("assume_timezone"):
            if "basis" not in spec:
                raise ConfigError(f"{where}: assume_timezone requires a 'basis' (documented with evidence, "
                                  "or assumption with assumption_ids).")
            just = BasisSpec.parse(where, spec)
            if just.basis is TransformBasis.STRUCTURAL:
                raise ConfigError(f"{where}: localising naive times is not structural; use 'documented' or 'assumption'.")
        elif any(k in spec for k in ("basis", "evidence", "assumption_ids")):
            raise ConfigError(f"{where}: basis/evidence/assumption_ids only apply together with assume_timezone.")
        return cls(_aliases(f"{where}.column", spec["column"]), spec["format"], spec.get("assume_timezone"), just)


@dataclass(frozen=True)
class TableSpec:
    name: str
    glob: str | None
    path: str | None
    participant_id: ParticipantIdSpec
    read: ReadSpec
    timestamp: TimestampSpec | None
    observations: tuple[ObservationSpec, ...]
    events: tuple[EventSpec, ...]
    attributes: tuple[AttributeSpec, ...]

    @classmethod
    def parse(cls, where: str, spec: Mapping[str, Any]) -> "TableSpec":
        _check_keys(where, spec, {"name", "files", "participant_id", "read", "timestamp", "observations", "events",
                                  "attributes"}, {"name", "files", "participant_id"})
        w = f"{where}[{spec['name']}]"
        files = spec["files"]
        _check_keys(f"{w}.files", files, {"glob", "path"})
        if ("glob" in files) == ("path" in files):
            raise ConfigError(f"{w}.files: give exactly one of 'glob' or 'path'.")
        obs = tuple(ObservationSpec.parse(f"{w}.observations", o) for o in spec.get("observations") or [])
        evs = tuple(EventSpec.parse(f"{w}.events", e) for e in spec.get("events") or [])
        atts = tuple(AttributeSpec.parse(f"{w}.attributes", a) for a in spec.get("attributes") or [])
        if not (obs or evs or atts):
            raise ConfigError(f"{w}: declares no observations, events or attributes.")
        ts = TimestampSpec.parse(f"{w}.timestamp", spec["timestamp"]) if spec.get("timestamp") else None
        if (obs or evs) and ts is None:
            raise ConfigError(f"{w}: observations/events require a 'timestamp' spec.")
        return cls(
            name=spec["name"], glob=files.get("glob"), path=files.get("path"),
            participant_id=ParticipantIdSpec.parse(f"{w}.participant_id", spec["participant_id"]),
            read=ReadSpec.parse(f"{w}.read", spec.get("read")), timestamp=ts,
            observations=obs, events=evs, attributes=atts,
        )


@dataclass(frozen=True)
class TemporalSpec:
    output_timezone: str | None  # None = naive local clock time
    date_shifted: bool = False
    notes: str | None = None

    @classmethod
    def parse(cls, where: str, spec: Mapping[str, Any]) -> "TemporalSpec":
        _check_keys(where, spec, {"output_timezone", "date_shifted", "notes"}, {"output_timezone"})
        return cls(spec["output_timezone"], bool(spec.get("date_shifted", False)), spec.get("notes"))


@dataclass(frozen=True)
class SourceConfig:
    source: str
    version: str
    temporal: TemporalSpec
    tables: tuple[TableSpec, ...]
    annotations: AnnotationRegistry
    plugins: tuple[str, ...] = ()
    event_schema: str | None = None  # "module.path:function" returning a schema given tz
    description: str | None = None
    raw: Mapping[str, Any] = field(default_factory=dict, compare=False, repr=False)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "SourceConfig":
        _check_keys("source config", data,
                    {"source", "version", "description", "temporal", "annotations", "plugins", "event_schema", "tables"},
                    {"source", "version", "temporal", "tables"})
        if not re.fullmatch(r"[a-z][a-z0-9_]*", data["source"]):
            raise ConfigError("source must be lower_snake_case (it namespaces participant IDs).")
        tables = tuple(TableSpec.parse("tables", t) for t in data["tables"])
        names = [t.name for t in tables]
        if len(set(names)) != len(names):
            raise ConfigError("table names must be unique.")
        ev_domains: dict[str, tuple[str, ...]] = {}
        for t in tables:
            for e in t.events:
                if e.event_type in ev_domains and ev_domains[e.event_type] != e.domains:
                    raise ConfigError(f"event_type {e.event_type!r} is declared with different domains in two places.")
                ev_domains[e.event_type] = e.domains
        streams = [o.stream_id for t in tables for o in t.observations]
        if len(set(streams)) != len(streams):
            raise ConfigError("stream_id values must be unique across the source.")
        cfg = cls(
            source=data["source"], version=str(data["version"]),
            temporal=TemporalSpec.parse("temporal", data["temporal"]), tables=tables,
            annotations=AnnotationRegistry.from_mapping(data.get("annotations")),
            plugins=tuple(data.get("plugins") or ()), event_schema=data.get("event_schema"),
            description=data.get("description"), raw=data,
        )
        return cfg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "SourceConfig":
        with open(path, encoding="utf-8") as fh:
            return cls.from_mapping(yaml.safe_load(fh) or {})

    # -- references used by the config (validated against registries at ingest time)
    def referenced_flags(self) -> set[str]:
        codes: set[str] = set()
        for t in self.tables:
            for o in t.observations:
                codes |= set(o.flags) | {r.flag for r in o.rules}
                for tr in o.transforms:
                    codes |= set(tr.flags)
            for e in t.events:
                codes |= set(e.flags)
                if e.subtype:
                    codes.add(e.subtype.unmapped_flag)
                for a in e.attributes:
                    codes |= {r.flag for r in a.rules}
            for a in t.attributes:
                codes |= set(a.flags) | {r.flag for r in a.rules}
        return codes

    def event_domains(self) -> dict[str, tuple[str, ...]]:
        return {e.event_type: e.domains for t in self.tables for e in t.events}

    def referenced_assumptions(self) -> set[str]:
        ids: set[str] = set()
        for t in self.tables:
            if t.timestamp and t.timestamp.timezone_justification:
                ids |= set(t.timestamp.timezone_justification.assumption_ids)
            for o in t.observations:
                if o.scale:
                    ids |= set(o.scale.justification.assumption_ids)
                if o.sampling_origin:
                    ids |= set(o.sampling_origin.justification.assumption_ids)
                for tr in o.transforms:
                    ids |= set(tr.justification.assumption_ids)
        return ids
