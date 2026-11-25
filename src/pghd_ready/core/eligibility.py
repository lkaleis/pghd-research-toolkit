"""Generic eligibility ledger (M3): ordered, named criteria with exclusion tracking.

Eligibility here is an analytical bookkeeping step ("which candidates meet this task's stated criteria"),
not a clinical or fitness judgement. Criteria are evaluated in order; every candidate is accounted for:
it is either eligible or excluded by exactly one criterion (the first one it fails). A criterion value of
``<NA>`` means "could not be evaluated" and also excludes, with its own status, so nothing is assumed.
"""

from __future__ import annotations

from typing import Sequence

import pandas as pd

ELIGIBLE = "eligible"


def apply_criteria(candidates: pd.DataFrame, criteria: Sequence[tuple[str, pd.Series]]) -> pd.DataFrame:
    """Ledger: ``candidates`` plus one boolean column per criterion, ``status`` and ``excluded_by``.

    Each criterion is ``(name, series)`` aligned to ``candidates.index``: True passes, False fails,
    NA is not evaluable. ``status`` is ``eligible``, ``excluded`` or ``not_evaluable``.
    """
    names = [n for n, _ in criteria]
    if len(set(names)) != len(names):
        raise ValueError("Criterion names must be unique.")
    ledger = candidates.copy()
    status = pd.Series(ELIGIBLE, index=ledger.index, dtype=object)
    excluded_by = pd.Series(pd.NA, index=ledger.index, dtype=object)
    for name, s in criteria:
        s = pd.Series(s, index=ledger.index).astype("boolean")
        ledger[f"crit_{name}"] = s
        open_ = status == ELIGIBLE
        fail, na = open_ & (s == False).fillna(False), open_ & s.isna()  # noqa: E712
        status[fail], excluded_by[fail] = "excluded", name
        status[na], excluded_by[na] = "not_evaluable", name
    ledger["status"], ledger["excluded_by"] = status, excluded_by
    return ledger


def funnel(ledger: pd.DataFrame, criteria_order: Sequence[str], by: Sequence[str] = ()) -> pd.DataFrame:
    """Aggregate counts: candidates, then remaining after each criterion and excluded by it (per group)."""
    by = list(by)
    groups = ledger.groupby(by, sort=True) if by else [((), ledger)]
    rows = []
    for key, g in groups:
        key = key if isinstance(key, tuple) else (key,)
        remaining = len(g)
        base = dict(zip(by, key))
        rows.append({**base, "step": 0, "criterion": "candidates", "excluded": 0, "not_evaluable": 0,
                     "remaining": remaining})
        for i, name in enumerate(criteria_order, start=1):
            exc = int(((g["excluded_by"] == name) & (g["status"] == "excluded")).sum())
            ne = int(((g["excluded_by"] == name) & (g["status"] == "not_evaluable")).sum())
            remaining -= exc + ne
            rows.append({**base, "step": i, "criterion": name, "excluded": exc, "not_evaluable": ne,
                         "remaining": remaining})
        if remaining != int((g["status"] == ELIGIBLE).sum()):
            raise AssertionError("Funnel does not account for every candidate.")
    return pd.DataFrame(rows, columns=by + ["step", "criterion", "excluded", "not_evaluable", "remaining"])
