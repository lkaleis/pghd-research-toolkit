import pandas as pd
import pandera.errors as pe
import pytest

from pghd_ready.conditions.cardiometabolic.schemas import meal_event_schema
from pghd_ready.core.schemas import (
    event_schema,
    observation_schema,
    participant_attribute_schema,
    stream_schema,
)

from conftest import make_attributes, make_events, make_observations, make_streams

CASES = [
    (observation_schema, make_observations),
    (event_schema, make_events),
    (participant_attribute_schema, make_attributes),
    (stream_schema, make_streams),
]


@pytest.mark.parametrize("schema_fn,factory", CASES)
@pytest.mark.parametrize("tz", [None, "UTC"])
def test_valid_synthetic_tables_pass(schema_fn, factory, tz):
    schema_fn(tz).validate(factory(tz), lazy=True)


def test_temporal_policy_is_enforced():
    with pytest.raises(pe.SchemaErrors):
        observation_schema("UTC").validate(make_observations(None), lazy=True)
    with pytest.raises(pe.SchemaErrors):
        observation_schema(None).validate(make_observations("UTC"), lazy=True)


def test_silent_nulling_is_rejected():
    obs = make_observations()
    obs.loc[0, "value"] = None  # raw "100" exists, value nulled, no flag
    with pytest.raises(pe.SchemaErrors, match="silent nulling"):
        observation_schema().validate(obs, lazy=True)
    obs.at[0, "quality_flags"] = ["documented_error_code"]
    observation_schema().validate(obs, lazy=True)


def test_silent_nulling_rule_for_attributes():
    attrs = make_attributes()
    attrs.at[2, "quality_flags"] = []
    with pytest.raises(pe.SchemaErrors, match="silent nulling"):
        participant_attribute_schema().validate(attrs, lazy=True)


def test_participant_id_must_be_namespaced():
    obs = make_observations()
    obs["participant_id"] = "p1"
    with pytest.raises(pe.SchemaErrors, match="participant_id"):
        observation_schema().validate(obs, lazy=True)


def test_flag_format_enforced():
    obs = make_observations()
    obs.at[0, "quality_flags"] = "suspected_spike"  # string instead of list
    with pytest.raises(pe.SchemaErrors):
        observation_schema().validate(obs, lazy=True)
    obs.at[0, "quality_flags"] = ["Bad Code"]
    with pytest.raises(pe.SchemaErrors):
        observation_schema().validate(obs, lazy=True)


def test_strict_schema_rejects_unexpected_columns():
    obs = make_observations()
    obs["glucose_mmol"] = 5.5
    with pytest.raises(pe.SchemaErrors):
        observation_schema().validate(obs, lazy=True)


def test_event_ids_unique_and_stream_keys_unique():
    ev = make_events()
    ev.loc[1, "event_id"] = ev.loc[0, "event_id"]
    with pytest.raises(pe.SchemaErrors):
        event_schema().validate(ev, lazy=True)
    st = make_streams()
    st.loc[1, "stream_id"] = "cgm_a"
    with pytest.raises(pe.SchemaErrors):
        stream_schema().validate(st, lazy=True)


def test_schemas_do_not_reject_implausible_values():
    """Plausibility is quality assessment's job; schemas only check conformance."""
    obs = make_observations()
    obs.loc[0, "value"] = -5000.0
    observation_schema().validate(obs, lazy=True)


def test_meal_event_schema_extends_core_events():
    ev = make_events()
    for col in ["energy_kcal", "carbohydrate_g", "protein_g", "fat_g", "fiber_g", "sugar_g", "amount_consumed_raw"]:
        ev[col] = [450.0, None]
    ev["image_ref"] = ["photos/a.jpg", None]
    meal_event_schema().validate(ev, lazy=True)
    # core schema stays strict: nutrition columns are not part of the core events table
    with pytest.raises(pe.SchemaErrors):
        event_schema().validate(ev, lazy=True)
    assert meal_event_schema().metadata["attribute_units"]["energy_kcal"] == "kcal"
