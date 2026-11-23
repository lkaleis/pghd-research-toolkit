import pandas as pd
import pytest

from pghd_ready.core import units
from pghd_ready.core.concepts import (
    Concept,
    ConceptError,
    ConceptKind,
    ConceptRegistry,
    ValueType,
    check_table_concepts,
)

from conftest import REPO_ROOT, make_observations

CARDIO_YAML = REPO_ROOT / "configs" / "conditions" / "cardiometabolic.yaml"


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
    assert reg.get("sex").value_type is ValueType.CATEGORICAL
    # HbA1c deliberately has no mmol/mol conversion (non-linear; not to be guessed)
    with pytest.raises(ConceptError):
        reg.get("hba1c").factor_to_canonical("mmol/L")
    assert len(reg.concepts) < 40  # guard against registry sprawl


def test_check_table_concepts():
    reg = ConceptRegistry.from_yaml(CARDIO_YAML)
    obs = make_observations()
    assert check_table_concepts(obs, reg, ConceptKind.OBSERVATION) == []
    obs.loc[0, "unit"] = "mmol/L"
    obs.loc[1, "concept"] = "body_weight"
    obs.loc[2, "concept"] = "mystery"
    problems = check_table_concepts(obs, reg, ConceptKind.OBSERVATION)
    assert any("non-canonical" in p for p in problems)
    assert any("not a observation" in p for p in problems)
    assert any("unregistered concept 'mystery'" in p for p in problems)
