"""Generic event-to-measurement alignment (M3): time windows, neighbouring events and attribute joins.

Everything here is read-only and dataset-neutral. Observations are never resampled, interpolated or
imputed: a window contains exactly the source observations whose timestamps fall inside it, with their
sampling origin carried through unchanged. Which observations are admissible is always an explicit
:class:`~pghd_ready.core.sampling.InclusionPolicy`; there is no default.

Window convention: a window ``[start_s, end_s]`` is relative to the event's anchor time (``start_time``);
``closed`` says which ends are included (``"both"``, ``"left"``, ``"right"``, ``"neither"``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from . import sampling as _sampling

WINDOW_COLUMNS = ["event_id", "participant_id", "stream_id", "window", "timestamp", "offset_s", "value", "unit",
                  "sampling_origin", "source_file", "source_row"]


@dataclass(frozen=True)
class Window:
    name: str
    start_s: float
    end_s: float
    closed: str = "both"

    def __post_init__(self):
        if self.closed not in ("both", "left", "right", "neither"):
            raise ValueError(f"Window {self.name!r}: closed must be both, left, right or neither.")
        if not self.end_s >= self.start_s:
            raise ValueError(f"Window {self.name!r}: end must not precede start.")


def extract_windows(events: pd.DataFrame, observations: pd.DataFrame, stream_ids: Iterable[str],
                    windows: Iterable[Window], policy: _sampling.InclusionPolicy | str) -> pd.DataFrame:
    """Observations of each stream inside each window around each event (long format, one row per reading).

    ``events`` needs ``event_id``, ``participant_id`` and ``start_time``. Readings are matched on participant.
    ``policy`` is required and recorded per row as ``sampling_origin`` (native / interpolated / unknown).
    """
    stream_ids, windows = list(stream_ids), list(windows)
    obs = observations[observations["stream_id"].isin(stream_ids) & observations["timestamp"].notna()]
    obs = _sampling.select_observations(obs, policy)
    obs = obs.assign(sampling_origin=_sampling.origin_labels(obs["is_interpolated"]).to_numpy())
    obs = obs.sort_values(["participant_id", "stream_id", "timestamp", "source_row"], kind="mergesort")
    ev = events.loc[events["start_time"].notna(), ["event_id", "participant_id", "start_time"]]
    parts = []
    for (pid, sid), g in obs.groupby(["participant_id", "stream_id"], sort=True):
        e = ev[ev["participant_id"] == pid]
        if e.empty:
            continue
        t = g["timestamp"].to_numpy()
        for w in windows:
            lo = (e["start_time"] + pd.to_timedelta(w.start_s, unit="s")).to_numpy()
            hi = (e["start_time"] + pd.to_timedelta(w.end_s, unit="s")).to_numpy()
            i0 = np.searchsorted(t, lo, side="left" if w.closed in ("both", "left") else "right")
            i1 = np.searchsorted(t, hi, side="right" if w.closed in ("both", "right") else "left")
            for eid, anchor, a, b in zip(e["event_id"], e["start_time"], i0, i1):
                if b <= a:
                    continue
                sl = g.iloc[a:b]
                parts.append(pd.DataFrame({
                    "event_id": eid, "participant_id": pid, "stream_id": sid, "window": w.name,
                    "timestamp": sl["timestamp"].to_numpy(),
                    "offset_s": (sl["timestamp"] - anchor).dt.total_seconds().to_numpy(),
                    "value": sl["value"].to_numpy(), "unit": sl["unit"].to_numpy(),
                    "sampling_origin": sl["sampling_origin"].to_numpy(),
                    "source_file": sl["source_file"].to_numpy(), "source_row": sl["source_row"].to_numpy()}))
    if not parts:
        return pd.DataFrame(columns=WINDOW_COLUMNS)
    out = pd.concat(parts, ignore_index=True)[WINDOW_COLUMNS]
    out["source_row"] = out["source_row"].astype("Int64")
    return out.sort_values(["event_id", "stream_id", "window", "timestamp"], kind="mergesort").reset_index(drop=True)


def neighbouring_events(index_events: pd.DataFrame, other_events: pd.DataFrame, before_s: float,
                        after_s: float) -> pd.DataFrame:
    """For each index event: other events of the same participant in [t0 - before_s, t0 + after_s] (closed).

    The index event itself is never counted (matched on ``event_id``). Returns ``event_id``, ``n_before``,
    ``n_after`` (an event at exactly t0 counts as after), ``n_neighbours`` and ``nearest_abs_s`` (smallest
    absolute time difference to any other event of the participant, inside or outside the window).
    """
    rows = []
    others = other_events.loc[other_events["start_time"].notna(), ["event_id", "participant_id", "start_time"]]
    by_pid = {pid: g for pid, g in others.groupby("participant_id")}
    for r in index_events.itertuples(index=False):
        g = by_pid.get(r.participant_id)
        g = g[g["event_id"] != r.event_id] if g is not None else None
        if g is None or g.empty or pd.isna(r.start_time):
            rows.append({"event_id": r.event_id, "n_before": 0, "n_after": 0, "n_neighbours": 0,
                         "nearest_abs_s": np.nan})
            continue
        d = (g["start_time"] - r.start_time).dt.total_seconds()
        nb, na = int(((d >= -before_s) & (d < 0)).sum()), int(((d >= 0) & (d <= after_s)).sum())
        rows.append({"event_id": r.event_id, "n_before": nb, "n_after": na, "n_neighbours": nb + na,
                     "nearest_abs_s": float(d.abs().min())})
    return pd.DataFrame(rows, columns=["event_id", "n_before", "n_after", "n_neighbours", "nearest_abs_s"])


def attributes_wide(attributes: pd.DataFrame, concepts: Mapping[str, str], dataset: str | None = None) -> pd.DataFrame:
    """One row per participant, one column per requested concept, in exactly the requested unit.

    Values stored in any other unit are not converted or pooled: they are left empty and counted in
    ``attrs.unit_mismatch``. A concept recorded more than once for a participant (in the requested unit) is an
    error, because a single value cannot be chosen without an analytical decision.
    """
    a = attributes if dataset is None else attributes[attributes["source_dataset"] == dataset]
    a = a[a["concept"].isin(list(concepts))]
    out = pd.DataFrame({"participant_id": sorted(attributes["participant_id"].dropna().unique()
                                                 if dataset is None else
                                                 attributes.loc[attributes["source_dataset"] == dataset,
                                                                "participant_id"].dropna().unique())})
    mismatch = {}
    for concept, unit in concepts.items():
        rows = a[a["concept"] == concept]
        ok = rows[rows["unit"] == unit]
        mismatch[concept] = int(len(rows) - len(ok))
        dup = ok["participant_id"].duplicated(keep=False)
        if dup.any():
            raise ValueError(f"Attribute {concept!r} [{unit}] is recorded more than once for "
                             f"{ok.loc[dup, 'participant_id'].nunique()} participant(s); choose explicitly.")
        out[concept] = out["participant_id"].map(ok.set_index("participant_id")["value_num"])
    out.attrs["unit_mismatch"] = mismatch
    return out
