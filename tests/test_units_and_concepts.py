import re

import pandas as pd
import pytest

from pghd_ready.core import units
from pghd_ready.core.concepts import (
    PGHD_DOMAINS,
    Concept,
    ConceptError,
    ConceptKind,
    ConceptRegistry,
    ValueType,
    check_table_concepts,
)

from conftest import REPO_ROOT, make_observations

CARDIO_YAML = REPO_ROOT / "configs" / "conditions" / "cardiometabolic.yaml"
SHARED_YAML = REPO_ROOT / "configs" / "concepts" / "shared.yaml"


# ------------------------------------------------------------------ units
def test_same_dimension_conversions():
    assert units.convert(100.0, "[lb_av]", "kg") == pytest.approx(45.359237)
    assert units.convert(10.0, "[in_i]", "cm") == pytest.approx(25.4)
    assert units.convert(2.0, "h", "min") == pytest.approx(120.0)


def test_round_trip_preserves_values_and_nulls():
    s = pd.Series([150.0, None, 200.5])
    back = units.convert(units.convert(s, "[lb_av]", "kg"), "kg", "[lb_av]")
    pd.testing.assert_series_equal(back, s)


def test_cross_dimension_conversion_is_refused():
    # glucose mmol/L <-> mg/dL depends on the substance, so the core must not do it
    with pytest.raises(units.UnitError, match="substance-specific"):
        units.conversion_factor("mmol/L", "mg/dL")


def test_unknown_unit_is_refused():
    with pytest.raises(units.UnitError, match="Unknown unit"):
        units.get_unit("furlongs")


# ------------------------------------------------------------------ concepts
def test_concept_specific_conversion_applies_only_to_that_concept():
    glucose = Concept("g", ConceptKind.OBSERVATION, ValueType.NUMERIC, "d", "mg/dL", {"mmol/L": 18.016})
    chol = Concept("c", ConceptKind.ATTRIBUTE, ValueType.NUMERIC, "d", "mg/dL")
    assert glucose.to_canonical(5.0, "mmol/L") == pytest.approx(90.08)
    with pytest.raises(ConceptError, match="no defined conversion"):
        chol.to_canonical(5.0, "mmol/L")


def test_concept_falls_back_to_generic_conversion():
    weight = Concept("w", ConceptKind.ATTRIBUTE, ValueType.NUMERIC, "d", "kg")
    assert weight.to_canonical(220.0, "[lb_av]") == pytest.approx(99.79032)


def test_concept_definition_rules():
    with pytest.raises(ConceptError, match="canonical_unit"):
        Concept("x", ConceptKind.OBSERVATION, ValueType.NUMERIC, "d")
    with pytest.raises(ConceptError, match="categorical"):
        Concept("x", ConceptKind.ATTRIBUTE, ValueType.CATEGORICAL, "d", "kg")
    with pytest.raises(units.UnitError):
        Concept("x", ConceptKind.ATTRIBUTE, ValueType.NUMERIC, "d", "stone")
    with pytest.raises(ConceptError, match="positive"):
        Concept("x", ConceptKind.ATTRIBUTE, ValueType.NUMERIC, "d", "mg/dL", {"mmol/L": -1})


def test_registry_rejects_unknown_fields():
    with pytest.raises(ConceptError, match="unknown fields"):
        ConceptRegistry.from_mapping(
            {"concepts": {"x": {"kind": "attribute", "value_type": "categorical", "description": "d", "colour": "red"}}}
        )


def test_cardiometabolic_registry_loads_and_is_minimal():
    reg = ConceptRegistry.from_yaml(CARDIO_YAML)
    assert reg.get("glucose_interstitial").canonical_unit == "mg/dL"
    assert reg.get("glucose_interstitial").kind is ConceptKind.OBSERVATION
    # HbA1c is accepted in % (NGSP) and mmol/mol (IFCC), each stored as measured; no conversion is defined
    # (the relation is affine and must not be guessed). Other units remain rejected.
    h = reg.get("hba1c")
    assert h.canonical_unit == "%" and h.accepted_units == ("mmol/mol",)
    assert h.stored_unit("%") == "%" and h.stored_unit("mmol/mol") == "mmol/mol"
    assert h.factor_to_canonical("mmol/mol") == 1.0  # factor to the *stored* unit (itself)
    assert h.to_canonical(6.5, "%") == 6.5  # canonical unit passes through unchanged
    with pytest.raises(ConceptError, match="accepted as measured"):
        h.to_canonical(48.0, "mmol/mol")  # accepted non-canonical unit: never silently relabelled as %
    with pytest.raises(ConceptError):
        h.factor_to_canonical("mmol/L")
    assert len(reg.concepts) < 40  # guard against registry sprawl


def test_shared_registry_holds_general_concepts_only():
    shared = ConceptRegistry.from_yaml(SHARED_YAML)
    cardio = ConceptRegistry.from_yaml(CARDIO_YAML)
    assert shared.get("sex").value_type is ValueType.CATEGORICAL
    assert {"heart_rate", "body_weight", "step_count", "age"} <= set(shared.concepts)
    assert not set(shared.concepts) & set(cardio.concepts)  # split, not duplicated
    condition_terms = re.compile(r"glucose|insulin|cholesterol|hba1c|lipid|triglycer|diabet", re.IGNORECASE)
    text = SHARED_YAML.read_text()
    assert not condition_terms.search(text), condition_terms.findall(text)
    assert len(shared.concepts) < 40


def test_registries_merge_and_conflicts_fail():
    reg = ConceptRegistry.from_yaml_files([SHARED_YAML, CARDIO_YAML])
    assert {"heart_rate", "glucose_interstitial"} <= set(reg.concepts)
    assert ConceptRegistry.from_yaml_files([SHARED_YAML, SHARED_YAML]).concepts == ConceptRegistry.from_yaml(
        SHARED_YAML).concepts  # identical redefinition is fine
    clash = ConceptRegistry.from_mapping({"concepts": {"heart_rate": {
        "kind": "observation", "value_type": "numeric", "canonical_unit": "/min", "description": "different"}}})
    with pytest.raises(ConceptError, match="Conflicting"):
        reg.merge(clash)


def test_domain_tags_are_optional_many_to_many_and_validated():
    reg = ConceptRegistry.from_yaml_files([SHARED_YAML, CARDIO_YAML])
    assert reg.domains_of("glucose_interstitial") == ("physiological",)
    assert reg.domains_of("activity_energy_expenditure") == ("behavioural", "physiological")
    assert reg.domains_of("sex") == ()  # untagged is allowed
    assert set(PGHD_DOMAINS) == {"physiological", "behavioural", "medication", "patient_reported"}
    for bad, msg in ((["behavioral"], "unknown domain"), (["physiological", "physiological"], "duplicate"),
                     ("physiological", "list of strings")):
        with pytest.raises(ConceptError, match=msg):
            ConceptRegistry.from_mapping({"concepts": {"x": {
                "kind": "attribute", "value_type": "categorical", "description": "d", "domains": bad}}})


def test_check_table_concepts():
    reg = ConceptRegistry.from_yaml_files([SHARED_YAML, CARDIO_YAML])
    obs = make_observations()
    assert check_table_concepts(obs, reg, ConceptKind.OBSERVATION) == []
    obs.loc[0, "unit"] = "mmol/L"
    obs.loc[1, "concept"] = "body_weight"
    obs.loc[2, "concept"] = "mystery"
    problems = check_table_concepts(obs, reg, ConceptKind.OBSERVATION)
    assert any("non-canonical" in p for p in problems)
    assert any("not a observation" in p for p in problems)
    assert any("unregistered concept 'mystery'" in p for p in problems)


def test_check_table_concepts_accepts_both_hba1c_units():
    reg = ConceptRegistry.from_yaml(CARDIO_YAML)
    df = pd.DataFrame({"concept": ["hba1c"] * 3, "unit": ["%", "mmol/mol", "mmol/L"]})
    problems = check_table_concepts(df, reg, ConceptKind.ATTRIBUTE)
    assert problems == ["concept 'hba1c' stored in non-canonical unit(s) ['mmol/L']"]
