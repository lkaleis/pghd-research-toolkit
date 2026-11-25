"""M3 generic core: windows, neighbouring events, attribute joins and the eligibility ledger. Synthetic only."""

import numpy as np
import pandas as pd
import pytest

from pghd_ready.core import alignment as al
from pghd_ready.core import eligibility as el

T0 = pd.Timestamp("2024-01-01 08:00:00")


def _obs(pid, stream, minutes, values=None, interp=None):
    n = len(minutes)
    return pd.DataFrame({
        "participant_id": pid, "source_dataset": "syn", "stream_id": stream, "concept": "c",
        "timestamp": [T0 + pd.Timedelta(minutes=m) for m in minutes],
        "value": values if values is not None else [float(m) for m in minutes], "unit": "u",
        "is_interpolated": pd.array(interp if interp is not None else [pd.NA] * n, dtype="boolean"),
        "source_file": f"{pid}.csv", "source_row": pd.array(range(10, 10 + n), dtype="Int64")})


def _ev(eid, pid, minute):
    return {"event_id": eid, "participant_id": pid, "start_time": T0 + pd.Timedelta(minutes=minute)}


# ------------------------------------------------------------------ windows
def test_window_boundaries_and_rows_are_exact_source_observations():
    obs = _obs("p1", "s", [-16, -15, -1, 0, 60, 120, 121])
    ev = pd.DataFrame([_ev("e1", "p1", 0)])
    w = al.extract_windows(ev, obs, ["s"], [al.Window("base", -900, 0, closed="left"),
                                            al.Window("post", 0, 7200, closed="both")], "all")
    base, post = w[w.window == "base"], w[w.window == "post"]
    assert base["offset_s"].tolist() == [-900, -60] and post["offset_s"].tolist() == [0, 3600, 7200]
    merged = w.merge(obs, on=["source_file", "source_row"], suffixes=("", "_src"))
    assert len(merged) == len(w) and (merged["value"] == merged["value_src"]).all()      # no resampling
    assert (merged["timestamp"] == merged["timestamp_src"]).all()


@pytest.mark.parametrize("closed, expected", [("both", [0, 60]), ("left", [0]), ("right", [60]), ("neither", [])])
def test_window_closure(closed, expected):
    w = al.extract_windows(pd.DataFrame([_ev("e1", "p1", 0)]), _obs("p1", "s", [0, 30, 60]), ["s"],
                           [al.Window("w", 0, 3600, closed=closed)], "all")
    assert sorted(set(w["offset_s"]) & {0, 3600}) == [m * 60 for m in expected]


def test_events_without_readings_other_participants_and_streams_are_not_mixed():
    obs = pd.concat([_obs("p1", "s", [0, 1]), _obs("p2", "s", [0]), _obs("p1", "other", [0])], ignore_index=True)
    ev = pd.DataFrame([_ev("e1", "p1", 0), _ev("e3", "p3", 0)])
    w = al.extract_windows(ev, obs, ["s"], [al.Window("w", 0, 600)], "all")
    assert set(w["event_id"]) == {"e1"} and set(w["participant_id"]) == {"p1"} and set(w["stream_id"]) == {"s"}
    assert len(w) == 2


def test_sampling_origin_policy_is_explicit_and_origin_is_carried():
    obs = _obs("p1", "s", [0, 1, 2], interp=[False, True, pd.NA])
    ev = pd.DataFrame([_ev("e1", "p1", 0)])
    win = [al.Window("w", 0, 600)]
    assert al.extract_windows(ev, obs, ["s"], win, "all")["sampling_origin"].tolist() == ["native", "interpolated", "unknown"]
    assert al.extract_windows(ev, obs, ["s"], win, "native_and_unknown")["sampling_origin"].tolist() == ["native", "unknown"]
    assert al.extract_windows(ev, obs, ["s"], win, "native_only")["sampling_origin"].tolist() == ["native"]
    with pytest.raises(ValueError):
        al.extract_windows(ev, obs, ["s"], win, "whatever")
    with pytest.raises(TypeError):
        al.extract_windows(ev, obs, ["s"], win)  # no default policy


def test_inputs_unchanged_and_deterministic():
    obs = _obs("p1", "s", [3, 0, 2, 1]).sample(frac=1, random_state=1)
    ev = pd.DataFrame([_ev("e2", "p1", 2), _ev("e1", "p1", 0)])
    before = obs.copy(deep=True)
    a = al.extract_windows(ev, obs, ["s"], [al.Window("w", 0, 120)], "all")
    pd.testing.assert_frame_equal(obs, before)
    pd.testing.assert_frame_equal(a, al.extract_windows(ev.iloc[::-1], obs.iloc[::-1], ["s"], [al.Window("w", 0, 120)], "all"))


# ------------------------------------------------------------------ neighbouring events
def test_neighbouring_events_boundaries_and_self_exclusion():
    idx = pd.DataFrame([_ev("i", "p1", 0)])
    others = pd.DataFrame([_ev("i", "p1", 0), _ev("a", "p1", -60), _ev("b", "p1", -61), _ev("c", "p1", 0),
                           _ev("d", "p1", 120), _ev("e", "p1", 121), _ev("x", "p2", 1)])
    r = al.neighbouring_events(idx, others, before_s=3600, after_s=7200).iloc[0]
    assert (r.n_before, r.n_after, r.n_neighbours, r.nearest_abs_s) == (1, 2, 3, 0.0)   # a; c (same time), d


def test_no_neighbours():
    r = al.neighbouring_events(pd.DataFrame([_ev("i", "p1", 0)]), pd.DataFrame([_ev("i", "p1", 0)]), 60, 60).iloc[0]
    assert r.n_neighbours == 0 and np.isnan(r.nearest_abs_s)


# ------------------------------------------------------------------ attributes
def test_attributes_wide_uses_only_the_requested_unit_and_refuses_duplicates():
    attrs = pd.DataFrame({"participant_id": ["p1", "p2", "p3"], "source_dataset": "syn", "concept": "hba1c",
                          "unit": ["%", "mmol/mol", "%"], "value_num": [6.0, 42.0, 5.5]})
    w = al.attributes_wide(attrs, {"hba1c": "%"}, dataset="syn")
    v = w.set_index("participant_id")["hba1c"]
    assert (v["p1"], v["p3"]) == (6.0, 5.5) and np.isnan(v["p2"])          # mmol/mol value neither converted nor pooled
    assert w.attrs["unit_mismatch"] == {"hba1c": 1}
    dup = pd.concat([attrs, attrs.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="more than once"):
        al.attributes_wide(dup, {"hba1c": "%"})


# ------------------------------------------------------------------ eligibility ledger
def test_first_failing_criterion_excludes_and_na_is_not_evaluable():
    c = pd.DataFrame({"id": list("abcde"), "x": [1, 0, 1, 1, 0], "y": [1, 1, 0, None, 0]})
    led = el.apply_criteria(c, [("x_ok", c["x"] == 1), ("y_ok", c["y"].map({1: True, 0: False}))])
    assert led["status"].tolist() == ["eligible", "excluded", "excluded", "not_evaluable", "excluded"]
    assert led["excluded_by"].tolist()[1:] == ["x_ok", "y_ok", "y_ok", "x_ok"]
    f = el.funnel(led, ["x_ok", "y_ok"])
    assert f["remaining"].tolist() == [5, 3, 1] and f["excluded"].tolist() == [0, 2, 1]
    assert f["not_evaluable"].tolist() == [0, 0, 1]


def test_funnel_per_group_accounts_for_every_candidate_and_names_are_unique():
    c = pd.DataFrame({"g": ["a", "a", "b"], "ok": [True, False, True]})
    f = el.funnel(el.apply_criteria(c, [("ok", c["ok"])]), ["ok"], by=["g"])
    assert f[f.criterion == "ok"].set_index("g")["remaining"].to_dict() == {"a": 1, "b": 1}
    with pytest.raises(ValueError, match="unique"):
        el.apply_criteria(c, [("ok", c["ok"]), ("ok", c["ok"])])
