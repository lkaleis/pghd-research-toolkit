"""M2a consistency checks (first slice: non-HDL = TC - HDL). Synthetic data only; no CGMacros data is read."""

import copy
import re
from decimal import Decimal

import numpy as np
import pandas as pd
import pytest

from pghd_ready.core import consistency as cs

from conftest import REPO_ROOT

QUALITY = REPO_ROOT / "configs" / "quality"

# A fictional, fully "verified" reference and an approved copy of the rule: test fixtures only.
SYNTH_REF = {"references": {"REF-SYNTH": {
    "type": "definition", "citation": "Synthetic test reference (fictional).", "bibliographic_verified": True,
    "content_verified": True, "verified_by": "test", "verified_on": "2026-11-24"}}}
SYNTH_RULE = {"rules": {"T-NONHDL": {
    "version": "1.0.0", "title": "non-HDL = TC - HDL (test)", "type": "linear_derivation",
    "target": {"concept": "non_hdl_cholesterol", "unit": "mg/dL"},
    "operands": [{"concept": "total_cholesterol", "unit": "mg/dL", "coefficient": 1},
                 {"concept": "hdl_cholesterol", "unit": "mg/dL", "coefficient": -1}],
    "reference": {"id": "REF-SYNTH", "locator": "eq. 1"}, "rationale": "test",
    "technical_review": {"status": "technically_approved", "by": "test", "on": "2026-11-24"},
    "clinical_review": {"status": "not_reviewed"}}}}
SYNTH_APP = {"dataset": "syn", "assumptions": {"A-SYN-01": {"statement": "test assumption"}},
             "applications": [{"rule_id": "T-NONHDL", "rule_version": "1.0.0",
                               "run_authorization": {"status": "review_mode_authorized", "by": "test",
                                                     "on": "2026-11-24"},
                               "co_measurement": {"same_record": True, "same_measured_at_raw": True,
                                                  "assumption_id": "A-SYN-01"},
                               "resolution": {c: {"value": 1, "unit": "mg/dL", "assumption_id": "A-SYN-01"}
                                              for c in ("non_hdl_cholesterol", "total_cholesterol",
                                                        "hdl_cholesterol")}}]}


def _setup(rule=None, app=None, ref=None):
    refs = cs.references_from_mapping(ref or SYNTH_REF)
    r = cs.rules_from_mapping(rule or SYNTH_RULE)["T-NONHDL"]
    a = cs.applications_from_mapping(app or SYNTH_APP)[0]
    return r, a, refs


def _row(pid, concept, raw, row, unit="mg/dL", time="8:00:00 AM", value=None, dataset="syn", file="p.csv"):
    v = float(raw) if value is None and raw not in (None, "") and raw.replace(".", "", 1).lstrip("-").isdigit() else value
    return {"participant_id": f"{dataset}:{pid}", "source_dataset": dataset, "concept": concept,
            "value_num": np.nan if v is None else v, "value_text": None, "unit": unit, "raw_value": raw,
            "measured_at": pd.NaT, "measured_at_raw": time, "quality_flags": [], "source_file": file,
            "source_row": row, "source_column": concept}


def _panel(pid, tc, hdl, nonhdl, row, **kw):
    return [_row(pid, "total_cholesterol", tc, row, **kw), _row(pid, "hdl_cholesterol", hdl, row, **kw),
            _row(pid, "non_hdl_cholesterol", nonhdl, row, **kw)]


def _df(rows):
    df = pd.DataFrame(rows)
    df["source_row"] = df["source_row"].astype("Int64")
    return df


def _run(rows, **setup):
    r, a, refs = _setup(**setup)
    return cs.evaluate(r, a, refs, _df(rows))


def _cov(res):
    return res.coverage.iloc[0].to_dict()


# ------------------------------------------------------------------ shipped configuration
def test_shipped_rule_application_and_reference_load_and_rule_does_not_run():
    refs = cs.load_references(QUALITY / "references.yaml")
    rules = cs.load_rules(QUALITY / "rules" / "cardiometabolic.yaml")
    apps = cs.load_applications(QUALITY / "applications" / "cgmacros.yaml")
    rule = rules["CM-CONS-NONHDL-001"]
    cs.validate_rule(rule, refs)
    assert rule.target == cs.Term("non_hdl_cholesterol", "mg/dL")
    assert [(t.concept, t.unit, t.coefficient) for t in rule.operands] == [
        ("total_cholesterol", "mg/dL", Decimal(1)), ("hdl_cholesterol", "mg/dL", Decimal(-1))]
    assert rule.technical_review.status == "draft" and rule.clinical_review.status == "not_reviewed"
    assert refs[rule.reference_id].verified and "page II-7" in rule.reference_locator
    (app,) = apps
    assert app.run_authorization.status == "not_authorized"
    assert app.dataset == "cgmacros" and app.assumption_ids == ("A-CGMACROS-LIPID-01",)
    assert "assumption, not a verified dataset fact" in app.assumptions["A-CGMACROS-LIPID-01"].statement
    assert all(v == Decimal(1) for v in app.resolution.values()) and app.same_measured_at_raw
    with pytest.raises(cs.ConsistencyError, match="not technically approved"):
        cs.evaluate(rule, app, refs, _df(_panel("001", "200", "50", "150", 2, dataset="cgmacros")))
    # even a technically approved copy of the rule does not run on CGMacros without run authorisation
    approved = cs.DerivationRule(**{**rule.__dict__, "technical_review": cs.Review("technically_approved", "t", "d")})
    with pytest.raises(cs.ConsistencyError, match="not authorised to run on dataset 'cgmacros'"):
        cs.evaluate(approved, app, refs, _df(_panel("001", "200", "50", "150", 2, dataset="cgmacros")))


def test_generic_rule_file_holds_no_dataset_specific_assumptions():
    text = (QUALITY / "rules" / "cardiometabolic.yaml").read_text(encoding="utf-8")
    # no dataset name, assumption ID, declared resolution value or co-measurement key in the generic rule
    assert not re.search(r"cgmacros|A-[A-Z]+-|1 mg/dL|resolution:|same_measured_at|same_record|assumption_id",
                         text, re.IGNORECASE)
    app_text = (QUALITY / "applications" / "cgmacros.yaml").read_text(encoding="utf-8")
    assert "same fasting screening lipid" in app_text and "1 mg/dL resolution" in app_text


def test_shipped_rule_concepts_exist_in_registry():
    from pghd_ready.core.concepts import ConceptRegistry
    reg = ConceptRegistry.from_yaml_files([REPO_ROOT / "configs" / "concepts" / "shared.yaml",
                                           REPO_ROOT / "configs" / "conditions" / "cardiometabolic.yaml"])
    rules = cs.load_rules(QUALITY / "rules" / "cardiometabolic.yaml")
    cs.validate_rule(rules["CM-CONS-NONHDL-001"], cs.load_references(QUALITY / "references.yaml"), registry=reg)


# ------------------------------------------------------------------ consistent, boundary, discrepancy
def test_consistent_values_including_rounding_differences_of_one():
    rows = (_panel("001", "200", "50", "150", 2) + _panel("002", "200", "50", "151", 3)
            + _panel("003", "200", "50", "149", 4))
    res = _run(rows)
    assert res.findings.empty
    c = _cov(res)
    assert c["n_candidates"] == c["n_evaluated"] == c["n_consistent"] == 3 and c["n_discrepancies"] == 0


def test_difference_of_two_is_a_discrepancy_labelled_for_review():
    rows = _panel("001", "200", "50", "152", 2) + _panel("002", "200", "50", "148", 3)
    res = _run(rows)
    f = res.findings
    assert len(f) == 2 and _cov(res)["n_discrepancies"] == 2
    assert (f["finding_type"] == "inconsistent").all() and (f["family"] == "consistency").all()
    assert (f["finding_label"] == "consistency discrepancy - requires review (not a confirmed error)").all()
    assert f["message"].str.contains("requires review (not a confirmed error)", regex=False).all()
    assert not f["message"].str.contains("invalid|incorrect|is an error|is wrong", case=False).any()
    one = f[f["participant_id"] == "syn:001"].iloc[0]
    assert (one["expected_low"], one["expected_high"]) == ("149.0", "151.0")
    assert (one["reported_low"], one["reported_high"]) == ("151.5", "152.5")
    assert one["reported_raw"] == "152" and one["assumption_ids"] == '["A-SYN-01"]'


def test_interval_arithmetic_and_touching_boundary():
    lo, hi = cs.achievable_interval([(Decimal(1), Decimal(200), Decimal(1)), (Decimal(-1), Decimal(50), Decimal(1))])
    assert (lo, hi) == (Decimal("149.0"), Decimal("151.0"))
    assert cs.intervals_overlap((Decimal(151), Decimal(152)), (lo, hi))          # touching: consistent
    assert not cs.intervals_overlap((Decimal("151.01"), Decimal(152)), (lo, hi))
    # end-to-end boundary with a target that has no declared resolution (zero-width interval)
    app = copy.deepcopy(SYNTH_APP)
    del app["applications"][0]["resolution"]["non_hdl_cholesterol"]
    res = _run(_panel("001", "200", "50", "151", 2) + _panel("002", "200", "50", "151.5", 3), app=app)
    assert _cov(res)["n_consistent"] == 1 and res.findings["participant_id"].tolist() == ["syn:002"]


def test_finer_declared_resolution_narrows_the_interval():
    app = copy.deepcopy(SYNTH_APP)
    for r in app["applications"][0]["resolution"].values():
        r["value"] = 0.5
    rows = _panel("001", "200", "50", "150.5", 2) + _panel("002", "200", "50", "151", 3)
    res = _run(rows, app=app)
    assert _cov(res)["n_consistent"] == 1 and res.findings["participant_id"].tolist() == ["syn:002"]


# ------------------------------------------------------------------ not evaluated (never assumed, never converted)
@pytest.mark.parametrize("rows, reason", [
    (_panel("001", "200", "50", "150", 2)[:1] + _panel("001", "200", "50", "150", 2)[2:], "operand_missing"),
    (_panel("001", "200", "50", "150", 2) + [_row("001", "hdl_cholesterol", "51", 2)], "operand_ambiguous"),
    ([_row("001", "total_cholesterol", "200", 2), _row("001", "hdl_cholesterol", "50", 3),
      _row("001", "non_hdl_cholesterol", "150", 2)], "operand_missing"),                        # HDL in another record
    (_panel("001", "200", "50", "150", 2)[:2] + [_row("001", "non_hdl_cholesterol", "3.9", 2, unit="mmol/L")],
     "unit_mismatch"),
    ([_row("001", "total_cholesterol", "5.2", 2, unit="mmol/L")] + _panel("001", "200", "50", "150", 2)[1:],
     "operand_unit_mismatch"),
    (_panel("001", "200", "50", "", 2), "value_null"),
    (_panel("001", "200", "", "150", 2), "value_null"),
    (_panel("001", "200", "50", "150", 2)[:2] + [_row("001", "non_hdl_cholesterol", "150", 2, time="9:15:00 AM")],
     "co_measurement_unverified"),
    (_panel("001", "200", "50", "150", 2, time=None), "co_measurement_unverified"),
    (_panel("001", "200", "50", "150.4", 2), "precision_conflict"),
    (_panel("001", "200", "50.5", "150", 2), "precision_conflict"),
    (_panel("001", "200", "50", "150", 2)[:2] + [_row("001", "non_hdl_cholesterol", "150 (low)", 2, value=150.0)],
     "not_numeric"),
    (_panel("001", "200", "50", "150", 2)[:1] + [_row("001", "hdl_cholesterol", "50", 2, value=50.1)]
     + _panel("001", "200", "50", "150", 2)[2:], "unit_converted"),
])
def test_values_that_cannot_be_compared_are_not_evaluated(rows, reason):
    res = _run(rows)
    c = _cov(res)
    assert res.findings.empty and c["n_evaluated"] == 0
    assert c[f"n_not_evaluated_{reason}"] == 1
    assert c["n_candidates"] == 1


def test_missing_lineage_cannot_demonstrate_same_record():
    rows = _panel("001", "200", "50", "150", None)
    assert _cov(_run(rows))["n_not_evaluated_co_measurement_unverified"] == 1


def test_same_measured_at_requirement_can_be_off_but_same_record_cannot():
    app = copy.deepcopy(SYNTH_APP)
    app["applications"][0]["co_measurement"]["same_measured_at_raw"] = False
    rows = _panel("001", "200", "50", "150", 2)[:2] + [_row("001", "non_hdl_cholesterol", "150", 2, time=None)]
    assert _cov(_run(rows, app=app))["n_consistent"] == 1
    app["applications"][0]["co_measurement"]["same_record"] = False
    with pytest.raises(cs.ConsistencyError, match="same_record must be true"):
        cs.applications_from_mapping(app)


def test_other_datasets_and_concepts_are_ignored():
    rows = (_panel("001", "200", "50", "160", 2, dataset="other")
            + [_row("001", "body_weight", "80", 2)] + _panel("002", "200", "50", "150", 3))
    res = _run(rows)
    assert res.findings.empty and _cov(res)["n_candidates"] == 1


def test_coverage_accounts_for_every_candidate():
    rows = (_panel("001", "200", "50", "150", 2) + _panel("002", "200", "50", "155", 3)
            + _panel("003", "200", "", "150", 4) + _panel("004", "200", "50", "150.4", 5))
    c = _cov(_run(rows))
    not_eval = sum(v for k, v in c.items() if k.startswith("n_not_evaluated_"))
    assert c["n_candidates"] == c["n_evaluated"] + not_eval == 4
    assert c["n_evaluated"] == c["n_consistent"] + c["n_discrepancies"] and c["n_discrepancies"] == 1


# ------------------------------------------------------------------ preservation and determinism
def test_inputs_unchanged_and_results_deterministic():
    rows = _panel("002", "210", "40", "175", 3) + _panel("001", "200", "50", "152", 2)
    df = _df(rows)
    before = df.copy(deep=True)
    r, a, refs = _setup()
    first, second = cs.evaluate(r, a, refs, df), cs.evaluate(r, a, refs, df)
    pd.testing.assert_frame_equal(df, before)
    pd.testing.assert_frame_equal(first.findings, second.findings)
    pd.testing.assert_frame_equal(first.coverage, second.coverage)
    assert first.findings["finding_id"].is_monotonic_increasing and first.findings["finding_id"].is_unique
    shuffled = cs.evaluate(r, a, refs, df.iloc[::-1].reset_index(drop=True))
    pd.testing.assert_frame_equal(first.findings, shuffled.findings)
    assert list(first.findings.columns) == cs.FINDING_COLUMNS


# ------------------------------------------------------------------ gating and validation
def test_rule_gating():
    rule = copy.deepcopy(SYNTH_RULE)
    rule["rules"]["T-NONHDL"]["technical_review"] = {"status": "draft"}
    with pytest.raises(cs.ConsistencyError, match="not technically approved"):
        _run(_panel("001", "200", "50", "150", 2), rule=rule)
    rule["rules"]["T-NONHDL"]["technical_review"] = {"status": "technically_approved", "by": "t", "on": "d"}
    rule["rules"]["T-NONHDL"]["clinical_review"] = {"status": "clinically_rejected"}
    with pytest.raises(cs.ConsistencyError, match="clinically rejected"):
        _run(_panel("001", "200", "50", "150", 2), rule=rule)


def test_run_authorisation_is_required_and_recorded():
    for auth in ({"status": "not_authorized"}, ):
        app = copy.deepcopy(SYNTH_APP)
        app["applications"][0]["run_authorization"] = auth
        with pytest.raises(cs.ConsistencyError, match="not authorised to run"):
            _run(_panel("001", "200", "50", "150", 2), app=app)
    app = copy.deepcopy(SYNTH_APP)
    app["applications"][0]["run_authorization"] = {"status": "review_mode_authorized"}
    with pytest.raises(cs.ConsistencyError, match="'by' and 'on'"):
        cs.applications_from_mapping(app)
    app["applications"][0].pop("run_authorization")
    with pytest.raises(cs.ConsistencyError, match="run_authorization"):
        cs.applications_from_mapping(app)


def test_gates_hold_for_directly_constructed_objects():
    """evaluate() itself enforces every gate; bypassing the YAML loaders does not bypass them."""
    rule, app, refs = _setup()
    df = _df(_panel("001", "200", "50", "152", 2))
    for auth in (cs.Review("not_authorized"), cs.Review("review_mode_authorized"),
                 cs.Review("review_mode_authorized", by="owner"), cs.Review("anything", by="o", on="d")):
        with pytest.raises(cs.ConsistencyError, match="authoris"):
            cs.evaluate(rule, cs.Application(**{**app.__dict__, "run_authorization": auth}), refs, df)
    for tech in (cs.Review("draft", by="o", on="d"), cs.Review("technically_approved")):
        with pytest.raises(cs.ConsistencyError, match="not technically approved|'by' and 'on'"):
            cs.evaluate(cs.DerivationRule(**{**rule.__dict__, "technical_review": tech}), app, refs, df)
    unverified = {k: cs.Reference(**{**v.__dict__, "content_verified": False}) for k, v in refs.items()}
    with pytest.raises(cs.ConsistencyError, match="verified reference"):
        cs.evaluate(rule, app, unverified, df)
    res = cs.evaluate(rule, app, refs, df)  # all gates satisfied: runs, and output is review-only
    assert len(res.findings) == 1 and res.findings.iloc[0]["finding_label"] == cs.FINDING_LABEL


def test_technical_approval_requires_verified_reference_and_reviewer():
    ref = copy.deepcopy(SYNTH_REF)
    ref["references"]["REF-SYNTH"]["content_verified"] = False
    with pytest.raises(cs.ConsistencyError, match="verified reference"):
        _run(_panel("001", "200", "50", "150", 2), ref=ref)
    rule = copy.deepcopy(SYNTH_RULE)
    rule["rules"]["T-NONHDL"]["technical_review"] = {"status": "technically_approved"}
    with pytest.raises(cs.ConsistencyError, match="'by' and 'on'"):
        _run(_panel("001", "200", "50", "150", 2), rule=rule)


@pytest.mark.parametrize("mutate, match", [
    (lambda r: r.update(type="range"), "linear_derivation"),
    (lambda r: r.update(finding_type="implausible"), "finding_type"),
    (lambda r: r["target"].update(unit="mg/dl"), "Unknown unit"),
    (lambda r: r["operands"][0].update(coefficient=0), "must not be zero"),
    (lambda r: r.pop("reference"), "reference"),
    (lambda r: r["clinical_review"].update(status="ok"), "invalid status"),
])
def test_invalid_rules_are_rejected(mutate, match):
    rule = copy.deepcopy(SYNTH_RULE)
    mutate(rule["rules"]["T-NONHDL"])
    with pytest.raises(cs.ConsistencyError, match=match):
        cs.rules_from_mapping(rule)


def test_unknown_reference_unknown_assumption_and_mismatches_are_rejected():
    rule = copy.deepcopy(SYNTH_RULE)
    rule["rules"]["T-NONHDL"]["reference"]["id"] = "REF-NONE"
    with pytest.raises(cs.ConsistencyError, match="unknown reference"):
        _run(_panel("001", "200", "50", "150", 2), rule=rule)
    app = copy.deepcopy(SYNTH_APP)
    app["applications"][0]["co_measurement"]["assumption_id"] = "A-NONE"
    with pytest.raises(cs.ConsistencyError, match="unknown assumption"):
        cs.applications_from_mapping(app)
    app = copy.deepcopy(SYNTH_APP)
    app["applications"][0]["rule_version"] = "2.0.0"
    with pytest.raises(cs.ConsistencyError, match="Application is for"):
        _run(_panel("001", "200", "50", "150", 2), app=app)
    app = copy.deepcopy(SYNTH_APP)
    app["applications"][0]["resolution"]["hdl_cholesterol"]["unit"] = "mmol/L"
    with pytest.raises(cs.ConsistencyError, match="no conversion is applied"):
        _run(_panel("001", "200", "50", "150", 2), app=app)
