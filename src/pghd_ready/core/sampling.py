"""Sampling origin of observations: the three-state ``is_interpolated`` column and inclusion policies.

``is_interpolated`` is a nullable boolean:

* ``False``  native: the reading is a sensor/device sample, either declared so by the source
  configuration (documented or assumption-based) or classified so by a named transform.
* ``True``   interpolated: the value was produced by interpolation, declared or classified.
* ``<NA>``   unknown: the default; the origin was not declared and could not be classified with
  sufficient confidence.

Whether a state was *declared* or *inferred* is recorded in the transformation log and in each row's
flags (classification transforms must add assumption-based flags); :func:`origin_evidence` reads it back
from a finished run. Classification transforms must
only assign ``False``/``True`` when their evidence is sufficient and leave ``<NA>`` otherwise, so
``native_only`` corresponds to "high-confidence native" readings.

Contract for classification transforms: add a declared, assumption-based flag to every reading whose
origin they set (or tried to set and left unknown), and leave readings they do not reclassify
untouched. In particular, readings with a declared origin that the transform does not reclassify must not
receive its flags, otherwise they would be counted as inferred.

Downstream code selects an explicit :class:`InclusionPolicy`; interpolated values are never counted as
measurements unless a policy says so. Running the same analysis under all three policies is a
sensitivity analysis.
"""

from __future__ import annotations

from enum import Enum

import pandas as pd

NATIVE, INTERPOLATED, UNKNOWN = "native", "interpolated", "unknown"
ORIGIN_LABELS = (NATIVE, INTERPOLATED, UNKNOWN)


class InclusionPolicy(str, Enum):
    NATIVE_ONLY = "native_only"  # is_interpolated == False
    NATIVE_AND_UNKNOWN = "native_and_unknown"  # is_interpolated is False or unknown
    ALL = "all"  # every reading, including interpolated values


def origin_labels(is_interpolated: pd.Series) -> pd.Series:
    """Map the nullable flag to 'native' / 'interpolated' / 'unknown'."""
    s = is_interpolated.astype("boolean")
    out = pd.Series(UNKNOWN, index=s.index, dtype=object)
    out[s.fillna(False).astype(bool)] = INTERPOLATED
    out[(~s).fillna(False).astype(bool)] = NATIVE
    return out


def select_observations(df: pd.DataFrame, policy: InclusionPolicy | str) -> pd.DataFrame:
    """Return the observations admitted by ``policy`` (never silently defaults)."""
    policy = InclusionPolicy(policy)
    labels = origin_labels(df["is_interpolated"])
    if policy is InclusionPolicy.NATIVE_ONLY:
        keep = labels == NATIVE
    elif policy is InclusionPolicy.NATIVE_AND_UNKNOWN:
        keep = labels != INTERPOLATED
    else:
        keep = pd.Series(True, index=df.index)
    return df[keep]


DECLARED, INFERRED = "declared", "inferred"


def _rec(r, key):
    return r.get(key) if isinstance(r, dict) else getattr(r, key)


def origin_evidence(observations: pd.DataFrame, log) -> pd.Series:
    """How each reading's origin was established: 'declared', 'inferred' or 'unknown'.

    Derived only from existing outputs, so it works on a finished run (``observations.parquet`` plus
    ``_provenance/transform_log.jsonl``), with no extra column:

    * **inferred**: the row carries a flag declared by a classification transform for its stream
      (transform log ``params.declared_flags``), i.e. the origin was inferred row by row;
    * **declared**: otherwise, the stream has a configuration-level ``sampling_origin`` declaration in
      the log (its basis, documented or assumption, is in the same log record);
    * **unknown**: ``is_interpolated`` is null.

    Combine with :func:`origin_labels` to separate e.g. declared-native from inferred-native readings, which
    differ in evidence strength although both are admitted by ``native_only``.
    """
    labels = origin_labels(observations["is_interpolated"])
    declared_streams, transform_flags = set(), {}
    basis = {}
    for r in log:
        op, stream = _rec(r, "operation"), _rec(r, "target")
        if op == "sampling_origin":
            declared_streams.add(stream)
            b = _rec(r, "basis")
            basis[stream] = getattr(b, "value", b)
        elif op == "transform":
            transform_flags.setdefault(stream, set()).update((_rec(r, "params") or {}).get("declared_flags", []))
    out = pd.Series(UNKNOWN, index=observations.index, dtype=object)
    for i, (lab, stream, flags) in enumerate(zip(labels, observations["stream_id"], observations["quality_flags"])):
        if lab == UNKNOWN:
            continue
        if set(flags) & transform_flags.get(stream, set()):
            out.iat[i] = INFERRED
        elif stream in declared_streams:
            out.iat[i] = DECLARED
        else:  # origin set without a recorded declaration or classification flag: report, never guess
            out.iat[i] = "unattributed"
    return out


def origin_evidence_counts(observations: pd.DataFrame, log) -> pd.DataFrame:
    """Per stream: native/interpolated readings split by declared vs inferred, plus unknown."""
    cols = ["native_declared", "native_inferred", "interpolated_declared", "interpolated_inferred", "unknown",
            "unattributed"]
    if not len(observations):
        return pd.DataFrame(columns=["stream_id"] + cols)
    lab = origin_labels(observations["is_interpolated"])
    ev = origin_evidence(observations, log)
    key = pd.Series([UNKNOWN if l == UNKNOWN else ("unattributed" if e == "unattributed" else f"{l}_{e}")
                     for l, e in zip(lab, ev)], index=observations.index)
    counts = pd.concat([observations["stream_id"], key.rename("k")], axis=1).groupby(["stream_id", "k"]).size()
    return counts.unstack("k", fill_value=0).reindex(columns=cols, fill_value=0).reset_index()


def origin_counts(df: pd.DataFrame, by: list[str] | tuple[str, ...] = ("stream_id",)) -> pd.DataFrame:
    """Counts of native / interpolated / unknown readings per group (all three columns always present)."""
    by = list(by)
    if not len(df):
        return pd.DataFrame(columns=by + list(ORIGIN_LABELS))
    labels = origin_labels(df["is_interpolated"]).rename("origin")
    counts = pd.concat([df[by], labels], axis=1).groupby(by + ["origin"]).size().unstack("origin", fill_value=0)
    return counts.reindex(columns=list(ORIGIN_LABELS), fill_value=0).reset_index()
