"""Definition-based consistency checks (M2a, first slice).

A *consistency* check asks whether reported values can all be true as reported,
given a definition linking them (for example a reported difference C = A - B)
and the resolution at which each value was reported. It is **not** a clinical
plausibility check: a finding means "these reported values contradict each other
and need review", never "this value is wrong" or "this value is abnormal".

Design (see the private M2 design specification):

* Generic **rules** (concept-level, no data) are kept separate from per-dataset
  **applications**, which declare the co-measurement requirement and the
  reporting resolution of each value, normally under an explicit assumption.
* Only rules whose ``technical_review`` status is ``technically_approved`` run,
  and technical approval requires a verified reference. ``clinically_rejected``
  rules never run. Technical approval is not a clinical judgement.
* A rule additionally runs on a dataset only when that dataset's application
  carries an explicit ``run_authorization`` (``review_mode_authorized``, with
  who and when). Rule review, dataset assumptions and run authorisation are
  separate records. Every assumption an application references must itself be
  approved (who and when) before the rule runs.
* Approvals are tied to the exact content reviewed. A technical approval must
  record ``content_sha256`` equal to :func:`rule_content_sha256` of the rule,
  and a run authorisation must record ``content_sha256`` equal to
  :func:`application_content_sha256` (which covers the application's
  evaluation-relevant content, its referenced assumptions and the rule's content
  hash). Any later change to the covered content invalidates the approval.
* Every operand must have a declared reporting resolution; an undeclared
  resolution is never treated as exact. (A target without one is treated as
  written, i.e. as a zero-width interval.)
* Rounding is handled with exact ``Decimal`` interval arithmetic: each value v
  reported at resolution r stands for [v - r/2, v + r/2]. A discrepancy is
  reported only when the reported target's interval cannot overlap the interval
  achievable from the operands. No other tolerance is used.
* Inputs are read, never modified. Values that cannot be compared safely (unit
  mismatch, missing or ambiguous operands, converted values, precision
  conflicts, unverified co-measurement) are counted as *not evaluated* with a
  reason; nothing is assumed or converted.

Only one rule type is implemented: ``linear_derivation`` (target = sum of
coefficient x operand) on the participant-attributes table.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from . import units as _units

FINDING_TYPE = "inconsistent"
FAMILY = "consistency"
FINDING_LABEL = "consistency discrepancy - requires review (not a confirmed error)"
KAHN = {"category": "conformance", "subcategory": "computational", "context": "verification"}

NOT_EVALUATED_REASONS = (
    "target_ambiguous",          # the target concept occurs more than once in the same record
    "unit_mismatch",             # target stored in a unit other than the rule's unit
    "value_null",                # target or operand value is empty (e.g. emptied by an M1 documented issue)
    "operand_missing",           # an operand concept is absent from the same record
    "operand_ambiguous",         # an operand concept occurs more than once in the same record
    "operand_unit_mismatch",     # an operand stored in a unit other than the rule's unit
    "co_measurement_unverified", # required co-measurement evidence (same measured_at_raw) is absent or differs
    "not_numeric",               # raw text cannot be read as a decimal number
    "unit_converted",            # stored value differs from the raw text (converted or scaled); precision unknown
    "precision_conflict",        # raw text is finer than the declared reporting resolution
)

FINDING_COLUMNS = [
    "finding_id", "rule_id", "rule_version", "family", "finding_type", "finding_label", "participant_id",
    "source_dataset", "source_file", "source_row", "target_concept", "unit", "reported_raw", "reported_low",
    "reported_high", "expected_low", "expected_high", "observed", "assumption_ids", "reference_id",
    "technical_review_status", "clinical_review_status", "message",
]
COVERAGE_COLUMNS = (["rule_id", "rule_version", "source_dataset", "n_candidates", "n_evaluated", "n_consistent",
                     "n_discrepancies"] + [f"n_not_evaluated_{r}" for r in NOT_EVALUATED_REASONS])


class ConsistencyError(ValueError):
    """Invalid rule, reference or application, or a rule that may not run."""


class TechnicalStatus(str, Enum):
    DRAFT = "draft"
    APPROVED = "technically_approved"
    RETIRED = "retired"


class ClinicalStatus(str, Enum):
    NOT_REVIEWED = "not_reviewed"
    REVIEWED = "clinically_reviewed"
    REJECTED = "clinically_rejected"


class RunAuthorization(str, Enum):
    """Owner authorisation to run a rule on a specific dataset (separate from rule review)."""
    NOT_AUTHORIZED = "not_authorized"
    REVIEW_MODE = "review_mode_authorized"


# ------------------------------------------------------------------ definitions
@dataclass(frozen=True)
class Reference:
    id: str
    type: str
    citation: str
    identifier: str | None
    version_or_date: str | None
    bibliographic_verified: bool
    content_verified: bool
    verified_by: str | None
    verified_on: str | None

    @property
    def verified(self) -> bool:
        return bool(self.bibliographic_verified and self.content_verified and self.verified_by and self.verified_on)


@dataclass(frozen=True)
class Term:
    concept: str
    unit: str
    coefficient: Decimal = Decimal(1)


@dataclass(frozen=True)
class Review:
    status: str
    by: str | None = None
    on: str | None = None
    notes: str | None = None
    content_sha256: str | None = None   # technical review / run authorisation only: hash of the content approved


@dataclass(frozen=True)
class DerivationRule:
    rule_id: str
    version: str
    title: str
    target: Term
    operands: tuple[Term, ...]
    reference_id: str
    reference_locator: str
    rationale: str
    limitations: tuple[str, ...]
    technical_review: Review
    clinical_review: Review

    def formula(self) -> str:
        parts = []
        for i, t in enumerate(self.operands):
            sign = "-" if t.coefficient < 0 else "+"
            mag = abs(t.coefficient)
            term = t.concept if mag == 1 else f"{mag} x {t.concept}"
            parts.append(term if i == 0 and sign == "+" else f"{sign} {term}")
        return " ".join(parts)


@dataclass(frozen=True)
class Assumption:
    id: str
    statement: str
    approved_by: str | None
    approved_on: str | None


@dataclass(frozen=True)
class Application:
    """How a rule is applied to one dataset: co-measurement requirement and declared resolutions."""
    dataset: str
    rule_id: str
    rule_version: str
    run_authorization: Review
    same_measured_at_raw: bool
    resolution: Mapping[str, Decimal]        # concept -> reporting resolution
    resolution_units: Mapping[str, str]      # concept -> unit of that resolution (must equal the rule's unit)
    resolution_assumptions: Mapping[str, str]  # concept -> ID of the assumption the declared resolution rests on
    co_measurement_assumption: str | None    # ID of the assumption the co-measurement declaration rests on
    assumptions: Mapping[str, Assumption]

    @property
    def assumption_ids(self) -> tuple[str, ...]:
        """All referenced assumption IDs, derived from the individual links (sorted, unique)."""
        ids = {a for a in self.resolution_assumptions.values() if a}
        if self.co_measurement_assumption:
            ids.add(self.co_measurement_assumption)
        return tuple(sorted(ids))


# ------------------------------------------------------------------ loading
def _req(spec: Mapping[str, Any], key: str, where: str) -> Any:
    if key not in spec or spec[key] in (None, ""):
        raise ConsistencyError(f"{where}: missing required field {key!r}.")
    return spec[key]


def _decimal(value: Any, where: str) -> Decimal:
    try:
        d = Decimal(str(value))
    except InvalidOperation as exc:
        raise ConsistencyError(f"{where}: {value!r} is not a number.") from exc
    if not d.is_finite():
        raise ConsistencyError(f"{where}: {value!r} is not finite.")
    return d


def references_from_mapping(data: Mapping[str, Any]) -> dict[str, Reference]:
    out = {}
    for rid, spec in (data.get("references") or {}).items():
        where = f"Reference {rid}"
        out[rid] = Reference(
            id=rid, type=_req(spec, "type", where), citation=_req(spec, "citation", where),
            identifier=spec.get("identifier"), version_or_date=spec.get("version_or_date"),
            bibliographic_verified=bool(spec.get("bibliographic_verified", False)),
            content_verified=bool(spec.get("content_verified", False)),
            verified_by=spec.get("verified_by"), verified_on=spec.get("verified_on"))
    return out


_SHA256 = re.compile(r"[0-9a-f]{64}")


def _review(spec: Mapping[str, Any], enum: type[Enum], where: str, hashed: bool = False) -> Review:
    status = _req(spec, "status", where)
    try:
        enum(status)
    except ValueError as exc:
        raise ConsistencyError(f"{where}: invalid status {status!r}; allowed: {[e.value for e in enum]}.") from exc
    digest = spec.get("content_sha256")
    if digest is not None:
        if not hashed:
            raise ConsistencyError(f"{where}: content_sha256 is not supported here.")
        if not (isinstance(digest, str) and _SHA256.fullmatch(digest)):
            raise ConsistencyError(f"{where}: content_sha256 must be 64 lowercase hexadecimal characters.")
    return Review(status=status, by=spec.get("by"), on=spec.get("on"), notes=spec.get("notes"),
                  content_sha256=digest)


def _term(spec: Mapping[str, Any], where: str, coefficient: bool) -> Term:
    unit = _req(spec, "unit", where)
    try:
        _units.get_unit(unit)
    except _units.UnitError as exc:
        raise ConsistencyError(f"{where}: {exc}") from exc
    coef = _decimal(_req(spec, "coefficient", where), where) if coefficient else Decimal(1)
    if coef == 0:
        raise ConsistencyError(f"{where}: coefficient must not be zero.")
    return Term(concept=_req(spec, "concept", where), unit=unit, coefficient=coef)


def _check_structure(rule: DerivationRule) -> None:
    """Structural validity of a linear derivation; applied on load and again before evaluation."""
    where = f"Rule {rule.rule_id}"
    if not rule.operands:
        raise ConsistencyError(f"{where}: at least one operand is required.")
    for t in rule.operands:
        if t.unit != rule.target.unit:
            raise ConsistencyError(f"{where}: operand {t.concept} is in {t.unit!r} but the target is in "
                                   f"{rule.target.unit!r}; all terms of a linear derivation must share one unit.")
    concepts = [t.concept for t in rule.operands]
    if rule.target.concept in concepts:
        raise ConsistencyError(f"{where}: the target concept {rule.target.concept!r} must not also be an operand.")
    if len(set(concepts)) != len(concepts):
        raise ConsistencyError(f"{where}: each operand concept may appear only once.")


def rules_from_mapping(data: Mapping[str, Any]) -> dict[str, DerivationRule]:
    out = {}
    for rid, spec in (data.get("rules") or {}).items():
        where = f"Rule {rid}"
        if _req(spec, "type", where) != "linear_derivation":
            raise ConsistencyError(f"{where}: only type 'linear_derivation' is supported.")
        if spec.get("finding_type", FINDING_TYPE) != FINDING_TYPE:
            raise ConsistencyError(f"{where}: a derivation rule produces finding_type {FINDING_TYPE!r} only.")
        operands = tuple(_term(o, f"{where} operand", True) for o in _req(spec, "operands", where))
        ref = _req(spec, "reference", where)
        rule = DerivationRule(
            rule_id=rid, version=str(_req(spec, "version", where)), title=_req(spec, "title", where),
            target=_term(_req(spec, "target", where), f"{where} target", False), operands=operands,
            reference_id=_req(ref, "id", f"{where} reference"),
            reference_locator=_req(ref, "locator", f"{where} reference"),
            rationale=_req(spec, "rationale", where), limitations=tuple(spec.get("limitations") or ()),
            technical_review=_review(_req(spec, "technical_review", where), TechnicalStatus,
                                     f"{where} technical_review", hashed=True),
            clinical_review=_review(_req(spec, "clinical_review", where), ClinicalStatus, f"{where} clinical_review"))
        _check_structure(rule)
        out[rid] = rule
    return out


def applications_from_mapping(data: Mapping[str, Any]) -> list[Application]:
    dataset = _req(data, "dataset", "Application file")
    assumptions = {aid: Assumption(id=aid, statement=_req(a, "statement", f"Assumption {aid}"),
                                   approved_by=a.get("approved_by"), approved_on=a.get("approved_on"))
                   for aid, a in (data.get("assumptions") or {}).items()}
    out = []
    for spec in data.get("applications") or []:
        where = f"Application of {spec.get('rule_id')!r} to {dataset}"
        cm = _req(spec, "co_measurement", where)
        if cm.get("same_record") is not True:
            raise ConsistencyError(f"{where}: co_measurement.same_record must be true (the only supported basis).")
        resolution, res_units, res_assumptions = {}, {}, {}
        for concept, r in (_req(spec, "resolution", where) or {}).items():
            value = _decimal(_req(r, "value", f"{where} resolution {concept}"), where)
            if value <= 0:
                raise ConsistencyError(f"{where}: resolution for {concept} must be positive.")
            resolution[concept] = value
            res_units[concept] = _req(r, "unit", f"{where} resolution {concept}")
            # every declaration must state its basis (no documented-source alternative is supported yet)
            res_assumptions[concept] = _req(r, "assumption_id", f"{where} resolution {concept}")
        cm_assumption = _req(cm, "assumption_id", f"{where} co_measurement")
        ids = set(res_assumptions.values()) | {cm_assumption}
        unknown = ids - set(assumptions)
        if unknown:
            raise ConsistencyError(f"{where}: unknown assumption(s) {sorted(unknown)}.")
        auth = _review(_req(spec, "run_authorization", where), RunAuthorization, f"{where} run_authorization",
                       hashed=True)
        if auth.status == RunAuthorization.REVIEW_MODE.value and not (auth.by and auth.on):
            raise ConsistencyError(f"{where}: run authorisation must record 'by' and 'on'.")
        out.append(Application(dataset=dataset, rule_id=_req(spec, "rule_id", where),
                               rule_version=str(_req(spec, "rule_version", where)), run_authorization=auth,
                               same_measured_at_raw=bool(cm.get("same_measured_at_raw", False)),
                               resolution=resolution, resolution_units=res_units,
                               resolution_assumptions=res_assumptions, co_measurement_assumption=cm_assumption,
                               assumptions=assumptions))
    return out


def _yaml(path: str | Path) -> Mapping[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def load_references(path: str | Path) -> dict[str, Reference]:
    return references_from_mapping(_yaml(path))


def load_rules(path: str | Path) -> dict[str, DerivationRule]:
    return rules_from_mapping(_yaml(path))


def load_applications(path: str | Path) -> list[Application]:
    return applications_from_mapping(_yaml(path))


# ------------------------------------------------------------------ content hashes (approval binding)
def _canonical_decimal(d: Decimal) -> str:
    """Value-based text for a Decimal, so 1, 1.0 and 1.00 hash identically."""
    return format(d.normalize(), "f")


def _sha256_json(payload: Mapping[str, Any]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def rule_content_sha256(rule: DerivationRule) -> str:
    """SHA-256 of a rule's reviewed content: everything except its two review records.

    Computed from the loaded rule (not the YAML text), so comments and formatting do not affect it.
    Operand order is part of the content.
    """
    return _sha256_json({
        "schema": "pghd_ready.consistency.rule/1",
        "rule_id": rule.rule_id, "version": rule.version, "type": "linear_derivation", "finding_type": FINDING_TYPE,
        "title": rule.title,
        "target": {"concept": rule.target.concept, "unit": rule.target.unit},
        "operands": [{"concept": t.concept, "unit": t.unit, "coefficient": _canonical_decimal(t.coefficient)}
                     for t in rule.operands],
        "reference": {"id": rule.reference_id, "locator": rule.reference_locator},
        "rationale": rule.rationale, "limitations": list(rule.limitations),
    })


def application_content_sha256(application: Application, rule: DerivationRule) -> str:
    """SHA-256 of everything a run authorisation approves: the application's evaluation-relevant content,
    the content of each assumption it references, and the content hash of the rule it applies.

    Excludes the run-authorisation record itself and assumptions the application does not reference.
    """
    assumptions = {}
    for aid in sorted(application.assumption_ids):
        a = application.assumptions.get(aid)
        if a is None:
            raise ConsistencyError(f"Application of {application.rule_id} to {application.dataset!r} references "
                                   f"unknown assumption {aid!r}.")
        assumptions[aid] = {"statement": a.statement, "approved_by": a.approved_by, "approved_on": a.approved_on}
    return _sha256_json({
        "schema": "pghd_ready.consistency.application/1",
        "dataset": application.dataset, "rule_id": application.rule_id, "rule_version": application.rule_version,
        "rule_content_sha256": rule_content_sha256(rule),
        "co_measurement": {"same_record": True, "same_measured_at_raw": bool(application.same_measured_at_raw),
                           "assumption_id": application.co_measurement_assumption},
        "resolution": {c: {"value": _canonical_decimal(v), "unit": application.resolution_units.get(c),
                           "assumption_id": application.resolution_assumptions.get(c)}
                       for c, v in application.resolution.items()},
        "assumptions": assumptions,
    })


def validate_rule(rule: DerivationRule, references: Mapping[str, Reference], registry=None) -> None:
    """Static checks; raises ConsistencyError. ``registry`` (optional) checks that concepts exist."""
    _check_structure(rule)
    ref = references.get(rule.reference_id)
    if ref is None:
        raise ConsistencyError(f"Rule {rule.rule_id}: unknown reference {rule.reference_id!r}.")
    if rule.technical_review.status == TechnicalStatus.APPROVED.value:
        if not ref.verified:
            raise ConsistencyError(f"Rule {rule.rule_id}: technical approval requires a verified reference "
                                   f"({rule.reference_id} is not fully verified).")
        if not (rule.technical_review.by and rule.technical_review.on):
            raise ConsistencyError(f"Rule {rule.rule_id}: technical approval must record 'by' and 'on'.")
        if not rule.technical_review.content_sha256:
            raise ConsistencyError(f"Rule {rule.rule_id}: technical approval must record content_sha256 "
                                   f"(see rule_content_sha256()).")
        if rule.technical_review.content_sha256 != rule_content_sha256(rule):
            raise ConsistencyError(f"Rule {rule.rule_id}: technical approval content_sha256 does not match the "
                                   f"current rule content; the rule changed after review and must be re-reviewed.")
    if registry is not None:
        for t in (rule.target, *rule.operands):
            if t.concept not in registry:
                raise ConsistencyError(f"Rule {rule.rule_id}: unknown concept {t.concept!r}.")


# ------------------------------------------------------------------ evaluation
@dataclass
class ConsistencyResult:
    findings: pd.DataFrame
    coverage: pd.DataFrame


def _raw_decimal(raw: Any) -> Decimal | None:
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    try:
        d = Decimal(str(raw).strip())
    except InvalidOperation:
        return None
    return d if d.is_finite() else None


def _interval(value: Decimal, resolution: Decimal | None) -> tuple[Decimal, Decimal]:
    half = resolution / 2 if resolution is not None else Decimal(0)
    return value - half, value + half


def achievable_interval(terms: list[tuple[Decimal, Decimal, Decimal | None]]) -> tuple[Decimal, Decimal]:
    """Exact interval of sum(coef * x) where each x lies in [v - r/2, v + r/2]. Items: (coef, value, resolution)."""
    lo = hi = Decimal(0)
    for coef, value, res in terms:
        a, b = _interval(value, res)
        lo += min(coef * a, coef * b)
        hi += max(coef * a, coef * b)
    return lo, hi


def intervals_overlap(a: tuple[Decimal, Decimal], b: tuple[Decimal, Decimal]) -> bool:
    """Closed intervals: touching at a boundary counts as overlapping (consistent)."""
    return a[0] <= b[1] and b[0] <= a[1]


def _finding_id(*parts: Any) -> str:
    return hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:16]


def evaluate(rule: DerivationRule, application: Application, references: Mapping[str, Reference],
             attributes: pd.DataFrame) -> ConsistencyResult:
    """Evaluate one technically approved derivation rule on a participant-attributes table (read-only)."""
    if application.rule_id != rule.rule_id or application.rule_version != rule.version:
        raise ConsistencyError(f"Application is for {application.rule_id} v{application.rule_version}, "
                               f"not {rule.rule_id} v{rule.version}.")
    validate_rule(rule, references)
    for t in (rule.target, *rule.operands):
        if t.concept in application.resolution and application.resolution_units.get(t.concept) != t.unit:
            raise ConsistencyError(f"Resolution for {t.concept} is declared in "
                                   f"{application.resolution_units.get(t.concept)!r}, "
                                   f"but the rule uses {t.unit!r}; no conversion is applied.")
    missing = [t.concept for t in rule.operands if t.concept not in application.resolution]
    if missing:  # an undeclared operand resolution is never treated as exact
        raise ConsistencyError(f"Application of {rule.rule_id} to {application.dataset!r} declares no reporting "
                               f"resolution for operand(s) {missing}; every operand needs one.")
    extra = sorted(set(application.resolution) - {rule.target.concept, *(t.concept for t in rule.operands)})
    if extra:
        raise ConsistencyError(f"Application of {rule.rule_id} to {application.dataset!r} declares resolution for "
                               f"concept(s) {extra} that the rule does not use.")
    unlinked = sorted(c for c in application.resolution if not application.resolution_assumptions.get(c))
    if unlinked:
        raise ConsistencyError(f"Application of {rule.rule_id} to {application.dataset!r}: resolution for {unlinked} "
                               f"references no assumption; every declaration needs an approved basis.")
    stray = sorted(set(application.resolution_assumptions) - set(application.resolution))
    if stray:
        raise ConsistencyError(f"Application of {rule.rule_id} to {application.dataset!r}: assumption links for "
                               f"{stray} have no resolution declaration.")
    if not application.co_measurement_assumption:
        raise ConsistencyError(f"Application of {rule.rule_id} to {application.dataset!r}: the co-measurement "
                               f"declaration references no assumption; every declaration needs an approved basis.")
    for aid in application.assumption_ids:
        a = application.assumptions.get(aid)
        if a is None:
            raise ConsistencyError(f"Application of {rule.rule_id} to {application.dataset!r} references unknown "
                                   f"assumption {aid!r}.")
        if not (a.approved_by and a.approved_on):
            raise ConsistencyError(f"Assumption {aid} is not approved (approved_by and approved_on are required) "
                                   f"and the rule does not run.")
    if rule.clinical_review.status == ClinicalStatus.REJECTED.value:
        raise ConsistencyError(f"Rule {rule.rule_id} is clinically rejected and never runs.")
    if rule.technical_review.status != TechnicalStatus.APPROVED.value:
        raise ConsistencyError(f"Rule {rule.rule_id} is not technically approved "
                               f"(status {rule.technical_review.status!r}); it does not run.")
    auth = application.run_authorization
    if auth.status != RunAuthorization.REVIEW_MODE.value:
        raise ConsistencyError(f"Rule {rule.rule_id} is not authorised to run on dataset {application.dataset!r} "
                               f"(run_authorization {auth.status!r}).")
    if not (auth.by and auth.on):  # enforced here too, so directly constructed Applications cannot bypass it
        raise ConsistencyError(f"Run authorisation for {rule.rule_id} on {application.dataset!r} must record 'by' and 'on'.")
    if not auth.content_sha256:
        raise ConsistencyError(f"Run authorisation for {rule.rule_id} on {application.dataset!r} must record "
                               f"content_sha256 (see application_content_sha256()).")
    if auth.content_sha256 != application_content_sha256(application, rule):
        raise ConsistencyError(f"Run authorisation content_sha256 for {rule.rule_id} on {application.dataset!r} does "
                               f"not match the current application, assumption or rule content; it changed after "
                               f"authorisation and must be re-authorised.")

    concepts = [rule.target.concept, *(t.concept for t in rule.operands)]
    cols = ["participant_id", "source_dataset", "concept", "value_num", "unit", "raw_value", "measured_at_raw",
            "source_file", "source_row"]
    sub = attributes.loc[(attributes["source_dataset"] == application.dataset)
                         & attributes["concept"].isin(concepts), cols].copy()
    key = ["participant_id", "source_file", "source_row"]
    by_record = {k: g for k, g in sub.groupby(key, sort=True, dropna=False)}
    targets = sub[sub["concept"] == rule.target.concept].sort_values(key, kind="mergesort").reset_index(drop=True)
    # a target repeated within one fully identified record cannot be attributed to one value: not evaluated
    ambiguous = (targets[key].notna().all(axis=1) & targets.duplicated(key, keep=False)).to_numpy()

    counts = {r: 0 for r in NOT_EVALUATED_REASONS}
    n_consistent = 0
    findings = []

    def check_row(row, term: Term) -> tuple[str | None, Decimal | None]:
        if row["unit"] != term.unit:
            return ("unit_mismatch" if term is rule.target else "operand_unit_mismatch"), None
        if pd.isna(row["value_num"]):
            return "value_null", None
        d = _raw_decimal(row["raw_value"])
        if d is None:
            return "not_numeric", None
        if float(d) != float(row["value_num"]):
            return "unit_converted", None
        res = application.resolution.get(term.concept)
        if res is not None and (d % res) != 0:
            return "precision_conflict", None
        return None, d

    for i, trow in targets.iterrows():
        if ambiguous[i]:
            reason, t_val = "target_ambiguous", None
        else:
            reason, t_val = check_row(trow, rule.target)
        if reason is None and any(pd.isna(trow[k]) for k in key):
            reason = "co_measurement_unverified"  # without full lineage, the same record cannot be demonstrated
        rec = by_record.get(tuple(trow[k] for k in key)) if reason is None else None
        operand_rows = []
        if reason is None:
            for term in rule.operands:
                rows = rec[rec["concept"] == term.concept]
                if len(rows) == 0:
                    reason = "operand_missing"
                    break
                if len(rows) > 1:
                    reason = "operand_ambiguous"
                    break
                operand_rows.append((term, rows.iloc[0]))
        if reason is None and application.same_measured_at_raw:
            times = [trow["measured_at_raw"], *(r["measured_at_raw"] for _, r in operand_rows)]
            texts = {str(t).strip() for t in times if not pd.isna(t) and str(t).strip()}
            if len(texts) != 1 or any(pd.isna(t) or not str(t).strip() for t in times):
                reason = "co_measurement_unverified"
        values = []
        if reason is None:
            for term, orow in operand_rows:
                r, d = check_row(orow, term)
                if r is not None:
                    reason = r
                    break
                values.append((term, orow, d))
        if reason is not None:
            counts[reason] += 1
            continue

        expected = achievable_interval([(t.coefficient, d, application.resolution.get(t.concept))
                                        for t, _, d in values])
        reported = _interval(t_val, application.resolution.get(rule.target.concept))
        if intervals_overlap(reported, expected):
            n_consistent += 1
            continue
        observed = {"target": {"concept": rule.target.concept, "raw_value": str(trow["raw_value"]),
                               "unit": trow["unit"]},
                    "operands": [{"concept": t.concept, "coefficient": str(t.coefficient),
                                  "raw_value": str(o["raw_value"]), "unit": o["unit"]} for t, o, _ in values]}
        findings.append({
            "finding_id": _finding_id(rule.rule_id, rule.version, application.dataset, trow["participant_id"],
                                      trow["source_file"], trow["source_row"]),
            "rule_id": rule.rule_id, "rule_version": rule.version, "family": FAMILY, "finding_type": FINDING_TYPE,
            "finding_label": FINDING_LABEL, "participant_id": trow["participant_id"],
            "source_dataset": application.dataset, "source_file": trow["source_file"],
            "source_row": trow["source_row"], "target_concept": rule.target.concept, "unit": rule.target.unit,
            "reported_raw": str(trow["raw_value"]), "reported_low": str(reported[0]), "reported_high": str(reported[1]),
            "expected_low": str(expected[0]), "expected_high": str(expected[1]),
            "observed": json.dumps(observed, sort_keys=True), "assumption_ids": json.dumps(list(application.assumption_ids)),
            "reference_id": rule.reference_id, "technical_review_status": rule.technical_review.status,
            "clinical_review_status": rule.clinical_review.status,
            "message": (f"Reported {rule.target.concept} cannot be reproduced from {rule.formula()} at the declared "
                        f"reporting resolution; {FINDING_LABEL}."),
        })

    fdf = pd.DataFrame(findings, columns=FINDING_COLUMNS)
    if len(fdf):
        fdf = fdf.sort_values("finding_id", kind="mergesort").reset_index(drop=True)
    fdf["source_row"] = fdf["source_row"].astype("Int64")
    n_eval = n_consistent + len(findings)
    coverage = pd.DataFrame([{
        "rule_id": rule.rule_id, "rule_version": rule.version, "source_dataset": application.dataset,
        "n_candidates": len(targets), "n_evaluated": n_eval, "n_consistent": n_consistent,
        "n_discrepancies": len(findings), **{f"n_not_evaluated_{r}": n for r, n in counts.items()},
    }], columns=COVERAGE_COLUMNS)
    return ConsistencyResult(findings=fdf, coverage=coverage)
