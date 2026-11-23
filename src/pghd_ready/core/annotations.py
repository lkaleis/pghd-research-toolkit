"""Annotation model for data issues and assumptions.

The toolkit distinguishes four kinds of annotation. Each has its own permitted
handling, so that ambiguous clinical values are never silently inferred or
corrected:

* ``documented_issue``: the source's own documentation states that a value is
  erroneous (e.g. a documented error code). Requires an ``evidence`` reference.
  The only category that may null a value. The original value is always kept
  in ``raw_value``.
* ``suspected_anomaly``: something looks wrong but is not documented as such
  (e.g. an implausible value). Flag only; the value is never changed.
* ``semantic_ambiguity``: the meaning of a field or value is unclear (e.g. an
  undocumented column, a unit that contradicts the data). Flag only; no
  interpretation is applied.
* ``assumption_based``: a value was produced or selected by a transformation
  that relies on an explicit, registered :class:`Assumption`. Flag only, and it
  must reference that assumption.

Flag codes appear in the ``quality_flags`` column of the common tables. Their
definitions live in an :class:`AnnotationRegistry`, normally loaded from YAML
configuration, so that every code in the data can be traced to a category and
a justification.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Mapping

import pandas as pd

FLAG_CODE_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")
ASSUMPTION_ID_PATTERN = re.compile(r"^[A-Z][A-Z0-9_-]*$")


class IssueCategory(str, Enum):
    DOCUMENTED_ISSUE = "documented_issue"
    SUSPECTED_ANOMALY = "suspected_anomaly"
    SEMANTIC_AMBIGUITY = "semantic_ambiguity"
    ASSUMPTION_BASED = "assumption_based"


class FlagAction(str, Enum):
    """What may happen to a value carrying the flag."""

    FLAG = "flag"  # annotate only; value unchanged
    SET_NULL = "set_null"  # value set to null; original preserved in raw_value


ALLOWED_ACTIONS: dict[IssueCategory, frozenset[FlagAction]] = {
    IssueCategory.DOCUMENTED_ISSUE: frozenset({FlagAction.FLAG, FlagAction.SET_NULL}),
    IssueCategory.SUSPECTED_ANOMALY: frozenset({FlagAction.FLAG}),
    IssueCategory.SEMANTIC_AMBIGUITY: frozenset({FlagAction.FLAG}),
    IssueCategory.ASSUMPTION_BASED: frozenset({FlagAction.FLAG}),
}


class AnnotationError(ValueError):
    """Raised when an annotation definition or usage violates the policy."""


@dataclass(frozen=True)
class Assumption:
    """An explicit assumption that a transformation relies on."""

    id: str
    statement: str
    rationale: str
    evidence: str | None = None

    def __post_init__(self) -> None:
        if not ASSUMPTION_ID_PATTERN.match(self.id):
            raise AnnotationError(f"Invalid assumption id {self.id!r}; expected e.g. 'A-CGM-01'.")
        if not self.statement.strip() or not self.rationale.strip():
            raise AnnotationError(f"Assumption {self.id} needs a non-empty statement and rationale.")


@dataclass(frozen=True)
class FlagDefinition:
    code: str
    category: IssueCategory
    description: str
    action: FlagAction = FlagAction.FLAG
    evidence: str | None = None
    assumption_id: str | None = None

    def __post_init__(self) -> None:
        if not FLAG_CODE_PATTERN.match(self.code):
            raise AnnotationError(f"Invalid flag code {self.code!r}; use lower_snake_case.")
        if not self.description.strip():
            raise AnnotationError(f"Flag {self.code} needs a description.")
        if self.action not in ALLOWED_ACTIONS[self.category]:
            raise AnnotationError(
                f"Flag {self.code}: action {self.action.value!r} is not permitted for category "
                f"{self.category.value!r} (allowed: {sorted(a.value for a in ALLOWED_ACTIONS[self.category])})."
            )
        if self.category is IssueCategory.DOCUMENTED_ISSUE and not self.evidence:
            raise AnnotationError(f"Flag {self.code}: documented issues must cite evidence.")
        if self.category is IssueCategory.ASSUMPTION_BASED and not self.assumption_id:
            raise AnnotationError(f"Flag {self.code}: assumption-based flags must reference an assumption_id.")
        if self.category is not IssueCategory.ASSUMPTION_BASED and self.assumption_id:
            raise AnnotationError(f"Flag {self.code}: only assumption-based flags may reference an assumption.")


@dataclass
class AnnotationRegistry:
    flags: dict[str, FlagDefinition] = field(default_factory=dict)
    assumptions: dict[str, Assumption] = field(default_factory=dict)

    # ---- construction -------------------------------------------------
    @classmethod
    def from_mapping(cls, data: Mapping[str, Any] | None) -> "AnnotationRegistry":
        """Build from a config mapping with optional ``assumptions`` and ``flags`` sections.

        ``assumptions: {A-ID: {statement, rationale, evidence?}}``
        ``flags: {code: {category, description, action?, evidence?, assumption_id?}}``
        """
        data = data or {}
        reg = cls()
        for aid, spec in (data.get("assumptions") or {}).items():
            reg.add_assumption(Assumption(id=aid, **spec))
        for code, spec in (data.get("flags") or {}).items():
            spec = dict(spec)
            try:
                category = IssueCategory(spec.pop("category"))
                action = FlagAction(spec.pop("action", FlagAction.FLAG.value))
            except (KeyError, ValueError) as exc:
                raise AnnotationError(f"Flag {code}: invalid or missing category/action ({exc}).") from exc
            reg.add_flag(FlagDefinition(code=code, category=category, action=action, **spec))
        return reg

    def add_assumption(self, assumption: Assumption) -> None:
        existing = self.assumptions.get(assumption.id)
        if existing is not None and existing != assumption:
            raise AnnotationError(f"Conflicting definitions for assumption {assumption.id}.")
        self.assumptions[assumption.id] = assumption

    def add_flag(self, flag: FlagDefinition) -> None:
        if flag.assumption_id and flag.assumption_id not in self.assumptions:
            raise AnnotationError(f"Flag {flag.code} references unknown assumption {flag.assumption_id}.")
        existing = self.flags.get(flag.code)
        if existing is not None and existing != flag:
            raise AnnotationError(f"Conflicting definitions for flag {flag.code}.")
        self.flags[flag.code] = flag

    def merge(self, other: "AnnotationRegistry") -> "AnnotationRegistry":
        """Return a new registry combining both; conflicting definitions raise."""
        merged = AnnotationRegistry(dict(self.flags), dict(self.assumptions))
        for a in other.assumptions.values():
            merged.add_assumption(a)
        for f in other.flags.values():
            merged.add_flag(f)
        return merged

    # ---- queries ------------------------------------------------------
    def category_of(self, code: str) -> IssueCategory:
        try:
            return self.flags[code].category
        except KeyError as exc:
            raise AnnotationError(f"Unregistered flag code {code!r}.") from exc

    def unknown_codes(self, codes: Iterable[str]) -> set[str]:
        return {c for c in codes if c not in self.flags}

    def to_dict(self) -> dict[str, Any]:
        """Serialisable snapshot (for run manifests)."""
        return {
            "assumptions": {
                a.id: {"statement": a.statement, "rationale": a.rationale, "evidence": a.evidence}
                for a in sorted(self.assumptions.values(), key=lambda a: a.id)
            },
            "flags": {
                f.code: {
                    "category": f.category.value,
                    "description": f.description,
                    "action": f.action.value,
                    "evidence": f.evidence,
                    "assumption_id": f.assumption_id,
                }
                for f in sorted(self.flags.values(), key=lambda f: f.code)
            },
        }


def flag_codes_in(df: pd.DataFrame, column: str = "quality_flags") -> Counter:
    """Count flag codes used in a table's flag column."""
    counts: Counter = Counter()
    for flags in df[column]:
        counts.update(flags)
    return counts


def validate_flags(df: pd.DataFrame, registry: AnnotationRegistry, column: str = "quality_flags") -> None:
    """Raise if the table uses any flag code that is not registered."""
    unknown = registry.unknown_codes(flag_codes_in(df, column))
    if unknown:
        raise AnnotationError(f"Unregistered flag codes in data: {sorted(unknown)}")


def summarize_flags(df: pd.DataFrame, registry: AnnotationRegistry, column: str = "quality_flags") -> pd.DataFrame:
    """Aggregate flag counts by category and code (no record-level output)."""
    counts = flag_codes_in(df, column)
    rows = [
        {"category": registry.category_of(code).value, "code": code, "n_records": n}
        for code, n in counts.items()
    ]
    out = pd.DataFrame(rows, columns=["category", "code", "n_records"])
    return out.sort_values(["category", "code"], ignore_index=True)
