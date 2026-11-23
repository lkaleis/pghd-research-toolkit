"""Minimal, explicit unit table using UCUM codes.

Only units needed by the initial cardiometabolic use case are defined. Each unit
belongs to a dimension and has a factor to that dimension's base unit; generic
conversion is allowed only within one dimension.

Conversions that depend on the measured substance (e.g. mmol/L to mg/dL,
which depends on the analyte's molar mass) are *not* defined here. They are declared
per concept in condition configuration (see :mod:`pghd_ready.core.concepts`),
so that the core holds no clinical knowledge. Unknown units or cross-dimension
conversions raise; nothing is guessed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


class UnitError(ValueError):
    """Raised for unknown units or conversions that are not defined."""


@dataclass(frozen=True)
class UnitDef:
    code: str  # UCUM code
    dimension: str
    factor_to_base: float


_UNITS: tuple[UnitDef, ...] = (
    # mass concentration (base: mg/dL)
    UnitDef("mg/dL", "mass_concentration", 1.0),
    # substance concentration (base: mmol/L); no generic link to mass concentration
    UnitDef("mmol/L", "substance_concentration", 1.0),
    # arbitrary (international) units per volume, e.g. microIU/mL for some hormone assays
    UnitDef("u[IU]/mL", "arbitrary_concentration", 1.0),
    # mass (base: kg)
    UnitDef("kg", "mass", 1.0),
    UnitDef("g", "mass", 1e-3),
    UnitDef("[lb_av]", "mass", 0.45359237),
    # length (base: m)
    UnitDef("m", "length", 1.0),
    UnitDef("cm", "length", 0.01),
    UnitDef("[in_i]", "length", 0.0254),
    # energy (base: kcal)
    UnitDef("kcal", "energy", 1.0),
    # time (base: s)
    UnitDef("s", "time", 1.0),
    UnitDef("min", "time", 60.0),
    UnitDef("h", "time", 3600.0),
    UnitDef("a", "time", 365.25 * 86400.0),
    # rates and others
    UnitDef("/min", "frequency", 1.0),
    UnitDef("kg/m2", "area_density", 1.0),
    UnitDef("%", "percent", 1.0),
    UnitDef("1", "dimensionless", 1.0),
    UnitDef("{MET}", "metabolic_equivalent", 1.0),
    UnitDef("{steps}", "count", 1.0),
)

UNITS: dict[str, UnitDef] = {u.code: u for u in _UNITS}


def get_unit(code: str) -> UnitDef:
    try:
        return UNITS[code]
    except KeyError as exc:
        raise UnitError(f"Unknown unit {code!r}. Add it to pghd_ready.core.units if it is genuinely needed.") from exc


def is_known_unit(code: str) -> bool:
    return code in UNITS


def conversion_factor(from_unit: str, to_unit: str) -> float:
    """Multiplicative factor converting values in ``from_unit`` to ``to_unit`` (same dimension only)."""
    src, dst = get_unit(from_unit), get_unit(to_unit)
    if src.dimension != dst.dimension:
        raise UnitError(
            f"Cannot convert {from_unit!r} ({src.dimension}) to {to_unit!r} ({dst.dimension}) generically; "
            "substance-specific conversions must be declared on the concept."
        )
    return src.factor_to_base / dst.factor_to_base


def convert(values: pd.Series | np.ndarray | float, from_unit: str, to_unit: str):
    """Convert values between units of the same dimension. Nulls stay null."""
    return values * conversion_factor(from_unit, to_unit)
