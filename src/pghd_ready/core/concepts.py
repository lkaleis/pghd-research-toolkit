"""Lightweight concept registry.

A *concept* names what a value means (e.g. ``heart_rate``, ``body_weight``)
independently of any source column name. Concepts are defined
in condition configuration (YAML), not in code. The registry deliberately holds
only what the toolkit uses: kind, value type, canonical unit, optional
concept-specific unit conversions, optional accepted units (stored as measured,
never converted), a description and optional references to external standards.
It is not a terminology server.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from . import units as _units


class ConceptError(ValueError):
    pass


class ConceptKind(str, Enum):
    OBSERVATION = "observation"  # time-stamped measurements (observations table)
    ATTRIBUTE = "attribute"  # participant-level/baseline values (participant_attributes table)


class ValueType(str, Enum):
    NUMERIC = "numeric"
    ORDINAL = "ordinal"  # numeric codes with order but not interval meaning
    CATEGORICAL = "categorical"  # stored in value_text


# Conceptual PGHD domains. Descriptive, many-to-many metadata: a concept (or event type) may belong to
# several domains or to none (e.g. demographics). They are not architectural boundaries.
#
# The vocabulary is deliberately FIXED IN CODE (here), so that typos such as "behavioral" fail loudly.
# Adding a domain therefore requires a code change to this tuple (plus tests and documentation).
# Making the vocabulary configurable is deferred until a real need arises.
PGHD_DOMAINS = ("physiological", "behavioural", "medication", "patient_reported")


def validate_domains(where: str, domains) -> tuple[str, ...]:
    if domains is None:
        return ()
    if isinstance(domains, str) or not all(isinstance(d, str) for d in domains):
        raise ConceptError(f"{where}: domains must be a list of strings.")
    unknown = sorted(set(domains) - set(PGHD_DOMAINS))
    if unknown:
        raise ConceptError(f"{where}: unknown domain(s) {unknown}; allowed: {list(PGHD_DOMAINS)}.")
    if len(set(domains)) != len(domains):
        raise ConceptError(f"{where}: duplicate domains.")
    return tuple(domains)


@dataclass(frozen=True)
class Concept:
    id: str
    kind: ConceptKind
    value_type: ValueType
    description: str
    canonical_unit: str | None = None
    # unit code -> multiplicative factor to canonical unit (substance-specific conversions)
    unit_conversions: Mapping[str, float] = field(default_factory=dict)
    references: Mapping[str, str] = field(default_factory=dict)
    domains: tuple[str, ...] = ()
    # Other units in which this concept is legitimately measured and which are stored AS MEASURED, never converted
    # (for example HbA1c in % (NGSP) or mmol/mol (IFCC), whose relation is affine, not a multiplicative factor).
    accepted_units: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        validate_domains(f"Concept {self.id}", self.domains)
        if self.value_type is ValueType.CATEGORICAL and self.canonical_unit is not None:
            raise ConceptError(f"Concept {self.id}: categorical concepts have no unit.")
        if self.value_type is ValueType.NUMERIC and self.canonical_unit is None:
            raise ConceptError(f"Concept {self.id}: numeric concepts need a canonical_unit.")
        if self.canonical_unit is not None:
            _units.get_unit(self.canonical_unit)
        for unit in self.accepted_units:
            _units.get_unit(unit)
            if unit == self.canonical_unit or unit in self.unit_conversions:
                raise ConceptError(f"Concept {self.id}: accepted unit {unit!r} duplicates the canonical unit or a conversion.")
        for unit, factor in self.unit_conversions.items():
            _units.get_unit(unit)
            if not isinstance(factor, (int, float)) or factor <= 0:
                raise ConceptError(f"Concept {self.id}: conversion factor for {unit!r} must be a positive number.")

    def stored_unit(self, unit: str) -> str | None:
        """Unit in which a value reported in ``unit`` is stored: itself if accepted as measured, else the canonical unit."""
        return unit if unit in self.accepted_units else self.canonical_unit

    def factor_to_canonical(self, unit: str) -> float:
        """Factor converting ``unit`` to the stored unit (1.0 for accepted-as-measured units); raises if undefined."""
        if self.canonical_unit is None:
            raise ConceptError(f"Concept {self.id} has no unit.")
        if unit == self.canonical_unit or unit in self.accepted_units:
            return 1.0
        if unit in self.unit_conversions:
            return float(self.unit_conversions[unit])
        try:
            return _units.conversion_factor(unit, self.canonical_unit)
        except _units.UnitError as exc:
            raise ConceptError(f"Concept {self.id}: no defined conversion from {unit!r} to {self.canonical_unit!r}.") from exc

    def to_canonical(self, values, unit: str):
        """Convert values to the canonical unit.

        The canonical unit always passes through unchanged. Accepted (as-measured) non-canonical units raise
        ``ConceptError``: no conversion is defined for them, so they must never be relabelled as canonical.
        """
        if unit != self.canonical_unit and unit in self.accepted_units:
            raise ConceptError(f"Concept {self.id}: {unit!r} is accepted as measured; no conversion to "
                               f"{self.canonical_unit!r} is defined.")
        return values * self.factor_to_canonical(unit)


@dataclass
class ConceptRegistry:
    concepts: dict[str, Concept] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ConceptRegistry":
        reg = cls()
        for cid, spec in (data.get("concepts") or {}).items():
            spec = dict(spec)
            try:
                concept = Concept(
                    id=cid,
                    kind=ConceptKind(spec.pop("kind")),
                    value_type=ValueType(spec.pop("value_type")),
                    description=spec.pop("description"),
                    canonical_unit=spec.pop("canonical_unit", None),
                    unit_conversions=dict(spec.pop("unit_conversions", None) or {}),
                    accepted_units=tuple(spec.pop("accepted_units", None) or ()),
                    references=dict(spec.pop("references", None) or {}),
                    domains=validate_domains(f"Concept {cid}", spec.pop("domains", None)),
                )
            except ConceptError:
                raise
            except (KeyError, ValueError) as exc:
                raise ConceptError(f"Concept {cid}: invalid definition ({exc}).") from exc
            if spec:
                raise ConceptError(f"Concept {cid}: unknown fields {sorted(spec)}.")
            reg.add(concept)
        return reg

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ConceptRegistry":
        with open(path, encoding="utf-8") as fh:
            return cls.from_mapping(yaml.safe_load(fh) or {})

    @classmethod
    def from_yaml_files(cls, paths) -> "ConceptRegistry":
        """Load and merge several registries (e.g. shared + condition-specific).

        A concept ID defined in more than one file must have identical definitions; otherwise loading fails.
        """
        if isinstance(paths, (str, Path)):
            paths = [paths]
        reg = cls()
        for p in paths:
            reg = reg.merge(cls.from_yaml(p))
        return reg

    def merge(self, other: "ConceptRegistry") -> "ConceptRegistry":
        merged = ConceptRegistry(dict(self.concepts))
        for c in other.concepts.values():
            merged.add(c)
        return merged

    def domains_of(self, concept_id: str) -> tuple[str, ...]:
        return self.concepts[concept_id].domains if concept_id in self.concepts else ()

    def add(self, concept: Concept) -> None:
        if concept.id in self.concepts and self.concepts[concept.id] != concept:
            raise ConceptError(f"Conflicting definitions for concept {concept.id}.")
        self.concepts[concept.id] = concept

    def get(self, concept_id: str) -> Concept:
        try:
            return self.concepts[concept_id]
        except KeyError as exc:
            raise ConceptError(f"Unknown concept {concept_id!r}.") from exc

    def __contains__(self, concept_id: str) -> bool:
        return concept_id in self.concepts


def domain_record_counts(registry: "ConceptRegistry", observations: pd.DataFrame, attributes: pd.DataFrame,
                         events: pd.DataFrame, event_domains: dict[str, tuple[str, ...]]) -> list[dict]:
    """Records per PGHD domain and table. A record counts once under each of its domains; untagged
    records are reported under domain None."""
    rows = []
    for table, df, key, lookup in (
        ("observations", observations, "concept", registry.domains_of),
        ("participant_attributes", attributes, "concept", registry.domains_of),
        ("events", events, "event_type", lambda et: event_domains.get(et, ())),
    ):
        if not len(df):
            continue
        for value, n in df.groupby(key).size().items():
            for d in (lookup(value) or (None,)):
                rows.append({"domain": d, "table": table, key: value, "n": int(n)})
    return rows


def check_table_concepts(df: pd.DataFrame, registry: ConceptRegistry, kind: ConceptKind) -> list[str]:
    """Return problems with concept usage in a common-representation table.

    Checks that each concept is registered, of the right kind, and that numeric
    values are stored in the concept's canonical unit or in one of its accepted (as-measured) units.
    """
    problems: list[str] = []
    unit_col = "unit" if "unit" in df.columns else None
    for concept_id, group in df.groupby("concept", sort=True):
        if concept_id not in registry:
            problems.append(f"unregistered concept {concept_id!r}")
            continue
        concept = registry.get(concept_id)
        if concept.kind is not kind:
            problems.append(f"concept {concept_id!r} is a {concept.kind.value}, not a {kind.value}")
        if unit_col and concept.canonical_unit is not None:
            bad = set(group[unit_col].dropna().unique()) - {concept.canonical_unit, *concept.accepted_units}
            if bad:
                problems.append(f"concept {concept_id!r} stored in non-canonical unit(s) {sorted(bad)}")
    return problems
