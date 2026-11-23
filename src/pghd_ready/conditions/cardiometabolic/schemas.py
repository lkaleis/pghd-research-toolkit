"""Cardiometabolic extensions of the core schemas."""

from __future__ import annotations

import pandera.pandas as pa

from ...core.schemas import event_schema

# Units are fixed by column name; values must be converted to these units on ingestion.
MEAL_ATTRIBUTE_UNITS = {
    "energy_kcal": "kcal",
    "carbohydrate_g": "g",
    "protein_g": "g",
    "fat_g": "g",
    "fiber_g": "g",
    "sugar_g": "g",
}


def meal_event_schema(tz: str | None = None) -> pa.DataFrameSchema:
    """Events table with nutrition attributes.

    The table can hold meal events and related events (e.g. meal photos); the
    nutrition columns are nullable for event types that do not carry them.

    ``amount_consumed_raw`` is stored exactly as the source reports it, with no
    interpretation (its semantics may be ambiguous; see source annotations).
    """
    nutrient_cols = {name: pa.Column(float, nullable=True) for name in MEAL_ATTRIBUTE_UNITS}
    schema = event_schema(tz).add_columns(
        {
            **nutrient_cols,
            "amount_consumed_raw": pa.Column(float, nullable=True),
            "image_ref": pa.Column(str, nullable=True),
        }
    )
    schema.name = "meal_events"
    schema.metadata = {**(schema.metadata or {}), "attribute_units": MEAL_ATTRIBUTE_UNITS}
    return schema
