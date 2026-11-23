"""Lightweight concept registry.

A *concept* names what a value means (e.g. ``heart_rate``, ``body_weight``)
independently of any source column name. Concepts are defined
in condition configuration (YAML), not in code. The registry deliberately holds
only what the toolkit uses: kind, value type, canonical unit, optional
concept-specific unit conversions, a description and optional references to
external standards. It is not a terminology server.
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

    def __post_init__(self) -> None:
        if self.value_type is ValueType.CATEGORICAL and self.canonical_unit is not None:
            raise ConceptError(f"Concept {self.id}: categorical concepts have no unit.")
        if self.value_type is ValueType.NUMERIC and self.canonical_unit is None:
            raise ConceptError(f"Concept {self.id}: numeric concepts need a canonical_unit.")
        if self.canonical_unit is not None:
            _units.get_unit(self.canonical_unit)
        for unit, factor in self.unit_conversions.items():
            _units.get_unit(unit)
            if not isinstance(factor, (int, float)) or factor <= 0:
                raise ConceptError(f"Concept {self.id}: conversion factor for {unit!r} must be a positive number.")

    def factor_to_canonical(self, unit: str) -> float:
        """Factor converting ``unit`` to the canonical unit; raises if undefined."""
        if self.canonical_unit is None:
            raise ConceptError(f"Concept {self.id} has no unit.")
        if unit == self.canonical_unit:
            return 1.0
        if unit in self.unit_conversions:
            return float(self.unit_conversions[unit])
        try:
            return _units.conversion_factor(unit, self.canonical_unit)
        except _units.UnitError as exc:
            raise ConceptError(f"Concept {self.id}: no defined conversion from {unit!r} to {self.canonical_unit!r}.") from exc

    def to_canonical(self, values, unit: str):
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
                    references=dict(spec.pop("references", None) or {}),
                )
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


def check_table_concepts(df: pd.DataFrame, registry: ConceptRegistry, kind: ConceptKind) -> list[str]:
    """Return problems with concept usage in a common-representation table.

    Checks that each concept is registered, of the right kind, and that numeric
    values are stored in the concept's canonical unit.
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
            bad = set(group[unit_col].dropna().unique()) - {concept.canonical_unit}
            if bad:
                problems.append(f"concept {concept_id!r} stored in non-canonical unit(s) {sorted(bad)}")
    return problems
