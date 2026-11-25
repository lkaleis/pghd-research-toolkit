"""M2a consistency checks (first slice: non-HDL = TC - HDL). Synthetic data only; no CGMacros data is read."""

import copy
import dataclasses
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
SYNTH_APP = {"dataset": "syn", "assumptions": {"A-SYN-01": {"statement": "test assumption", "approved_by": "test",
                                                             "approved_on": "2026-11-24"}},
             "applications": [{"rule_id": "T-NONHDL", "rule_version": "1.0.0",
                               "run_authorization": {"status": "review_mode_authorized", "by": "test",
                                                     "on": "2026-11-24"},
                               "co_measurement": {"same_record": True, "same_measured_at_raw": True,
                                                  "assumption_id": "A-SYN-01"},
                               "resolution": {c: {"value": 1, "unit": "mg/dL", "assumption_id": "A-SYN-01"}
                                              for c in ("non_hdl_cholesterol", "total_cholesterol",
                                                        "hdl_cholesterol")}}]}


# Content hashes of the unmodified fixtures above. Pinned so that any change to the canonical hashing scheme
# (which would silently invalidate every recorded approval) fails here and is made deliberately.
RULE_FIXTURE_SHA256 = "450802c6ab2f9d355296a1fe238c8ea290868ee009885d0056b068a918dcc0ec"
APP_FIXTURE_SHA256 = "a8fdfba91234155238ab27aa6b2c17c8ed47e35b1c547d09d4b56aa85e3f663f"


def _stamp(rule, app):
    """Record content hashes matching the current content, as a reviewer would after reviewing it (tests only)."""
    rule = dataclasses.replace(rule, technical_review=dataclasses.replace(
        rule.technical_review, content_sha256=cs.rule_content_sha256(rule)))
    app = dataclasses.replace(app, run_authorization=dataclasses.replace(
        app.run_authorization, content_sha256=cs.application_content_sha256(app, rule)))
    return rule, app


def _setup(rule=None, app=None, ref=None, stamp=True):
    refs = cs.references_from_mapping(ref or SYNTH_REF)
    r = cs.rules_from_mapping(rule or SYNTH_RULE)["T-NONHDL"]
    a = cs.applications_from_mapping(app or SYNTH_APP)[0]
    if stamp:
        r, a = _stamp(r, a)
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
    approved = cs.DerivationRule(**{**rule.__dict__, "technical_review": cs.Review(
        "technically_approved", "t", "d", content_sha256=cs.rule_content_sha256(rule))})
    with pytest.raises(cs.ConsistencyError, match="not authorised to run on dataset 'cgmacros'"):
        cs.evaluate(approved, app, refs, _df(_panel("001", "200", "50", "150", 2, dataset="cgmacros")))
    # the shipped configuration records no approval hashes (approval is the owner's step)
    assert rule.technical_review.content_sha256 is None and app.run_authorization.content_sha256 is None


def test_shipped_rule_and_application_meet_the_new_safeguards():
    """The non-HDL rule and its CGMacros application satisfy D1, D3 and N1 unchanged."""
    rule = cs.load_rules(QUALITY / "rules" / "cardiometabolic.yaml")["CM-CONS-NONHDL-001"]
    (app,) = cs.load_applications(QUALITY / "applications" / "cgmacros.yaml")
    cs._check_structure(rule)
    assert {t.concept for t in rule.operands} <= set(app.resolution)
    assert set(app.resolution) <= {rule.target.concept, *(t.concept for t in rule.operands)}
    assert all(app.assumptions[a].approved_by and app.assumptions[a].approved_on for a in app.assumption_ids)
    # hashes are computable for the owner's review (values not asserted: they follow the shipped content)
    assert re.fullmatch(r"[0-9a-f]{64}", cs.rule_content_sha256(rule))
    assert re.fullmatch(r"[0-9a-f]{64}", cs.application_content_sha256(app, rule))


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


# ------------------------------------------------------------------ D1: every operand needs a declared resolution
@pytest.mark.parametrize("drop", [("total_cholesterol",), ("hdl_cholesterol",),
                                  ("non_hdl_cholesterol", "total_cholesterol", "hdl_cholesterol")])
def test_undeclared_operand_resolution_is_refused_not_treated_as_exact(drop):
    app = copy.deepcopy(SYNTH_APP)
    for c in drop:
        del app["applications"][0]["resolution"][c]
    with pytest.raises(cs.ConsistencyError, match="no reporting resolution for operand"):
        _run(_panel("001", "200", "50", "151", 2), app=app)


def test_resolution_for_an_unused_concept_is_refused():
    app = copy.deepcopy(SYNTH_APP)
    app["applications"][0]["resolution"]["ldl_cholesterol"] = {"value": 1, "unit": "mg/dL", "assumption_id": "A-SYN-01"}
    with pytest.raises(cs.ConsistencyError, match="does not use"):
        _run(_panel("001", "200", "50", "150", 2), app=app)


def test_undeclared_operand_resolution_is_refused_for_directly_constructed_application():
    rule, app, refs = _setup()
    res = {k: v for k, v in app.resolution.items() if k != "hdl_cholesterol"}
    bare = dataclasses.replace(app, resolution=res,
                               resolution_units={k: v for k, v in app.resolution_units.items() if k in res})
    _, bare = _stamp(rule, bare)  # even with a matching authorisation hash
    with pytest.raises(cs.ConsistencyError, match="no reporting resolution for operand"):
        cs.evaluate(rule, bare, refs, _df(_panel("001", "200", "50", "151", 2)))


# ------------------------------------------------------------------ D2: repeated target within one record
def test_target_repeated_in_one_record_is_not_evaluated():
    rows = _panel("001", "200", "50", "160", 2) + [_row("001", "non_hdl_cholesterol", "170", 2)]
    res = _run(rows)
    c = _cov(res)
    assert res.findings.empty
    assert c["n_candidates"] == 2 and c["n_not_evaluated_target_ambiguous"] == 2 and c["n_evaluated"] == 0
    assert c["n_candidates"] == c["n_evaluated"] + sum(v for k, v in c.items() if k.startswith("n_not_evaluated_"))


def test_target_ambiguity_is_per_record_and_finding_ids_stay_unique():
    rows = (_panel("001", "200", "50", "160", 2) + [_row("001", "non_hdl_cholesterol", "170", 2)]
            + _panel("001", "200", "50", "160", 3) + _panel("002", "200", "50", "160", 2))
    res = _run(rows)
    c = _cov(res)
    assert c["n_not_evaluated_target_ambiguous"] == 2 and c["n_discrepancies"] == 2
    assert res.findings["finding_id"].is_unique


# ------------------------------------------------------------------ D3: referenced assumptions must be approved
@pytest.mark.parametrize("approval", [{}, {"approved_by": "test"}, {"approved_on": "2026-11-24"}])
def test_unapproved_assumption_blocks_the_run(approval):
    app = copy.deepcopy(SYNTH_APP)
    app["assumptions"]["A-SYN-01"] = {"statement": "test assumption", **approval}
    with pytest.raises(cs.ConsistencyError, match="Assumption A-SYN-01 is not approved"):
        _run(_panel("001", "200", "50", "150", 2), app=app)


def test_assumption_gate_holds_for_directly_constructed_application():
    rule, app, refs = _setup()
    df = _df(_panel("001", "200", "50", "150", 2))
    unapproved = dataclasses.replace(app, assumptions={"A-SYN-01": cs.Assumption("A-SYN-01", "test", None, None)})
    with pytest.raises(cs.ConsistencyError, match="not approved"):
        cs.evaluate(rule, unapproved, refs, df)
    dangling = dataclasses.replace(app, assumptions={})
    with pytest.raises(cs.ConsistencyError, match="unknown assumption"):
        cs.evaluate(rule, dangling, refs, df)


# ------------------------------------------------------------------ N1: structural validity of a rule
STRUCTURE_CASES = [
    (lambda r: r.update(operands=[]), "at least one operand"),
    (lambda r: r["operands"][1].update(unit="mmol/L"), "must share one unit"),
    (lambda r: r["target"].update(unit="mmol/L"), "must share one unit"),
    (lambda r: r["operands"].append({"concept": "non_hdl_cholesterol", "unit": "mg/dL", "coefficient": 1}),
     "must not also be an operand"),
    (lambda r: r["operands"].append({"concept": "hdl_cholesterol", "unit": "mg/dL", "coefficient": 1}),
     "only once"),
]


@pytest.mark.parametrize("mutate, match", STRUCTURE_CASES)
def test_structurally_invalid_rules_are_rejected_on_load(mutate, match):
    rule = copy.deepcopy(SYNTH_RULE)
    mutate(rule["rules"]["T-NONHDL"])
    with pytest.raises(cs.ConsistencyError, match=match):
        cs.rules_from_mapping(rule)


@pytest.mark.parametrize("change, match", [
    ({"operands": ()}, "at least one operand"),
    ({"operands": (cs.Term("total_cholesterol", "mg/dL", Decimal(1)), cs.Term("hdl_cholesterol", "mmol/L", Decimal(-1)))},
     "must share one unit"),
    ({"operands": (cs.Term("total_cholesterol", "mg/dL", Decimal(1)), cs.Term("non_hdl_cholesterol", "mg/dL", Decimal(-1)))},
     "must not also be an operand"),
    ({"operands": (cs.Term("total_cholesterol", "mg/dL", Decimal(1)), cs.Term("total_cholesterol", "mg/dL", Decimal(-1)))},
     "only once"),
])
def test_structurally_invalid_rules_are_rejected_when_constructed_directly(change, match):
    rule, app, refs = _setup()
    bad = dataclasses.replace(rule, **change)
    bad, app = _stamp(bad, app)  # even with hashes matching the bad content
    with pytest.raises(cs.ConsistencyError, match=match):
        cs.evaluate(bad, app, refs, _df(_panel("001", "200", "50", "150", 2)))


# ------------------------------------------------------------------ N2: technical approval bound to rule content
def test_technical_approval_requires_a_matching_rule_content_hash():
    rule, app, refs = _setup()
    df = _df(_panel("001", "200", "50", "150", 2))
    no_hash = dataclasses.replace(rule, technical_review=dataclasses.replace(rule.technical_review,
                                                                             content_sha256=None))
    with pytest.raises(cs.ConsistencyError, match="must record content_sha256"):
        cs.evaluate(no_hash, app, refs, df)
    wrong = dataclasses.replace(rule, technical_review=dataclasses.replace(rule.technical_review,
                                                                           content_sha256="0" * 64))
    with pytest.raises(cs.ConsistencyError, match="does not match the current rule content"):
        cs.evaluate(wrong, app, refs, df)
    assert len(cs.evaluate(rule, app, refs, df).coverage) == 1  # the matching hash runs


@pytest.mark.parametrize("change", [
    {"version": "1.0.1"},
    {"title": "changed title"},
    {"target": cs.Term("non_hdl_cholesterol", "mmol/L")},
    {"operands": (cs.Term("total_cholesterol", "mg/dL", Decimal(1)), cs.Term("hdl_cholesterol", "mg/dL", Decimal("-0.9")))},
    {"operands": (cs.Term("total_cholesterol", "mg/dL", Decimal(1)), cs.Term("ldl_cholesterol", "mg/dL", Decimal(-1)))},
    {"operands": (cs.Term("hdl_cholesterol", "mg/dL", Decimal(-1)), cs.Term("total_cholesterol", "mg/dL", Decimal(1)))},
    {"reference_id": "REF-OTHER"},
    {"reference_locator": "eq. 2"},
    {"rationale": "different rationale"},
    {"limitations": ("a new limitation",)},
])
def test_any_change_to_reviewed_rule_content_invalidates_technical_approval(change):
    rule, _, _ = _setup()
    assert cs.rule_content_sha256(dataclasses.replace(rule, **change)) != cs.rule_content_sha256(rule)


def test_changed_rule_content_is_refused_by_evaluate_when_constructed_directly():
    rule, app, refs = _setup()
    edited = dataclasses.replace(rule, operands=(cs.Term("total_cholesterol", "mg/dL", Decimal(1)),
                                                 cs.Term("hdl_cholesterol", "mg/dL", Decimal("-0.9"))))
    with pytest.raises(cs.ConsistencyError, match="does not match the current rule content"):
        cs.evaluate(edited, app, refs, _df(_panel("001", "200", "50", "150", 2)))


def test_rule_hash_ignores_review_records_and_number_spelling_and_is_pinned():
    rule, _, _ = _setup(stamp=False)
    reviewed = dataclasses.replace(rule, clinical_review=cs.Review("clinically_reviewed", "x", "y"),
                                   technical_review=cs.Review("draft", notes="n"))
    assert cs.rule_content_sha256(reviewed) == cs.rule_content_sha256(rule)
    spelled = copy.deepcopy(SYNTH_RULE)
    spelled["rules"]["T-NONHDL"]["operands"][0]["coefficient"] = "1.00"
    spelled["rules"]["T-NONHDL"]["operands"][1]["coefficient"] = -1.0
    assert cs.rule_content_sha256(cs.rules_from_mapping(spelled)["T-NONHDL"]) == cs.rule_content_sha256(rule)
    # pins the canonical form: an accidental change to the hashing scheme must be a deliberate, visible change
    assert cs.rule_content_sha256(rule) == RULE_FIXTURE_SHA256


# ------------------------------------------------------------------ run authorisation bound to application content
def test_run_authorization_requires_a_matching_application_content_hash():
    rule, app, refs = _setup()
    df = _df(_panel("001", "200", "50", "150", 2))
    for digest, match in ((None, "must record content_sha256"), ("0" * 64, "must be re-authorised")):
        a = dataclasses.replace(app, run_authorization=dataclasses.replace(app.run_authorization,
                                                                           content_sha256=digest))
        with pytest.raises(cs.ConsistencyError, match=match):
            cs.evaluate(rule, a, refs, df)


@pytest.mark.parametrize("change", [
    {"same_measured_at_raw": False},
    {"resolution": {"non_hdl_cholesterol": Decimal("0.5"), "total_cholesterol": Decimal(1), "hdl_cholesterol": Decimal(1)}},
    {"resolution": {"total_cholesterol": Decimal(1), "hdl_cholesterol": Decimal(1)},        # target resolution removed
     "resolution_assumptions": {"total_cholesterol": "A-SYN-01", "hdl_cholesterol": "A-SYN-01"}},
    {"assumptions": {"A-SYN-01": cs.Assumption("A-SYN-01", "a different statement", "test", "2026-11-24")}},
    {"assumptions": {"A-SYN-01": cs.Assumption("A-SYN-01", "test assumption", "someone else", "2026-11-24")}},
    {"assumptions": {"A-SYN-01": cs.Assumption("A-SYN-01", "test assumption", "test", "2026-11-25")}},
    {"dataset": "syn2"},
])
def test_any_change_to_evaluation_relevant_application_content_invalidates_run_authorization(change):
    rule, app, refs = _setup()
    changed = dataclasses.replace(app, **change)
    assert cs.application_content_sha256(changed, rule) != cs.application_content_sha256(app, rule)
    rows = _panel("001", "200", "50", "150", 2, dataset=changed.dataset)
    with pytest.raises(cs.ConsistencyError, match="must be re-authorised"):
        cs.evaluate(rule, changed, refs, _df(rows))


def test_rule_change_invalidates_run_authorization_even_if_technical_approval_is_restamped():
    rule, app, refs = _setup()
    edited = dataclasses.replace(rule, rationale="edited after authorisation")
    edited = dataclasses.replace(edited, technical_review=dataclasses.replace(
        edited.technical_review, content_sha256=cs.rule_content_sha256(edited)))
    with pytest.raises(cs.ConsistencyError, match="must be re-authorised"):
        cs.evaluate(edited, app, refs, _df(_panel("001", "200", "50", "150", 2)))


def test_application_hash_ignores_authorization_record_and_unreferenced_assumptions():
    rule, app, _ = _setup()
    base = cs.application_content_sha256(app, rule)
    other = dataclasses.replace(app, run_authorization=cs.Review("not_authorized", notes="x"),
                                assumptions={**app.assumptions, "A-UNUSED": cs.Assumption("A-UNUSED", "s", None, None)})
    assert cs.application_content_sha256(other, rule) == base
    spelled = copy.deepcopy(SYNTH_APP)
    spelled["applications"][0]["resolution"]["total_cholesterol"]["value"] = "1.0"
    assert cs.application_content_sha256(cs.applications_from_mapping(spelled)[0], rule) == base
    assert base == APP_FIXTURE_SHA256


def test_content_hash_field_is_validated_and_limited_to_hashed_records():
    rule = copy.deepcopy(SYNTH_RULE)
    rule["rules"]["T-NONHDL"]["technical_review"]["content_sha256"] = "ABC"
    with pytest.raises(cs.ConsistencyError, match="64 lowercase hexadecimal"):
        cs.rules_from_mapping(rule)
    rule = copy.deepcopy(SYNTH_RULE)
    rule["rules"]["T-NONHDL"]["clinical_review"]["content_sha256"] = "0" * 64
    with pytest.raises(cs.ConsistencyError, match="not supported here"):
        cs.rules_from_mapping(rule)
    rule = copy.deepcopy(SYNTH_RULE)
    r = cs.rules_from_mapping(rule)["T-NONHDL"]
    rule["rules"]["T-NONHDL"]["technical_review"]["content_sha256"] = cs.rule_content_sha256(r)
    assert cs.rules_from_mapping(rule)["T-NONHDL"].technical_review.content_sha256 == cs.rule_content_sha256(r)


# ------------------------------------------------------------------ per-declaration assumption links
def _two_assumption_app():
    app = copy.deepcopy(SYNTH_APP)
    app["assumptions"]["A-SYN-02"] = {"statement": "second test assumption", "approved_by": "test",
                                      "approved_on": "2026-11-24"}
    return app


def test_each_declaration_keeps_its_own_assumption_link():
    app = _two_assumption_app()
    app["applications"][0]["resolution"]["hdl_cholesterol"]["assumption_id"] = "A-SYN-02"
    a = cs.applications_from_mapping(app)[0]
    assert a.resolution_assumptions == {"non_hdl_cholesterol": "A-SYN-01", "total_cholesterol": "A-SYN-01",
                                        "hdl_cholesterol": "A-SYN-02"}
    assert a.co_measurement_assumption == "A-SYN-01" and a.assumption_ids == ("A-SYN-01", "A-SYN-02")


def test_shipped_cgmacros_declarations_all_rest_on_the_lipid_assumption():
    (app,) = cs.load_applications(QUALITY / "applications" / "cgmacros.yaml")
    assert app.co_measurement_assumption == "A-CGMACROS-LIPID-01"
    assert app.resolution_assumptions == {c: "A-CGMACROS-LIPID-01" for c in app.resolution}


@pytest.mark.parametrize("where", ["non_hdl_cholesterol", "total_cholesterol", "hdl_cholesterol", "co_measurement"])
def test_declaration_without_an_assumption_is_refused_on_load(where):
    app = copy.deepcopy(SYNTH_APP)
    spec = app["applications"][0]
    (spec["co_measurement"] if where == "co_measurement" else spec["resolution"][where]).pop("assumption_id")
    with pytest.raises(cs.ConsistencyError, match="assumption_id"):
        cs.applications_from_mapping(app)


@pytest.mark.parametrize("change, match", [
    ({"resolution_assumptions": {"non_hdl_cholesterol": "A-SYN-01", "total_cholesterol": "A-SYN-01"}},
     r"resolution for \['hdl_cholesterol'\] references no assumption"),
    ({"resolution_assumptions": {"non_hdl_cholesterol": "A-SYN-01", "total_cholesterol": "A-SYN-01",
                                 "hdl_cholesterol": ""}}, "references no assumption"),
    ({"resolution_assumptions": {"non_hdl_cholesterol": "A-SYN-01", "total_cholesterol": "A-SYN-01",
                                 "hdl_cholesterol": "A-SYN-01", "ldl_cholesterol": "A-SYN-01"}},
     "have no resolution declaration"),
    ({"co_measurement_assumption": None}, "co-measurement declaration references no assumption"),
    ({"co_measurement_assumption": "A-NONE"}, "unknown assumption"),
])
def test_missing_or_dangling_links_are_refused_when_constructed_directly(change, match):
    rule, app, refs = _setup()
    bad = dataclasses.replace(app, **change)
    if "A-NONE" not in str(change):
        _, bad = _stamp(rule, bad)  # even with an authorisation hash matching the bad content
    with pytest.raises(cs.ConsistencyError, match=match):
        cs.evaluate(rule, bad, refs, _df(_panel("001", "200", "50", "150", 2)))


def test_unapproved_assumption_linked_from_a_single_declaration_blocks_the_run():
    app = _two_assumption_app()
    app["assumptions"]["A-SYN-02"] = {"statement": "second test assumption"}   # not approved
    app["applications"][0]["resolution"]["total_cholesterol"]["assumption_id"] = "A-SYN-02"
    with pytest.raises(cs.ConsistencyError, match="Assumption A-SYN-02 is not approved"):
        _run(_panel("001", "200", "50", "150", 2), app=app)


def _app_hash(app_mapping, rule):
    return cs.application_content_sha256(cs.applications_from_mapping(app_mapping)[0], rule)


def test_moving_or_swapping_an_assumption_link_changes_the_application_hash():
    rule, _, _ = _setup()
    base = _two_assumption_app()
    hdl, tc, cm = copy.deepcopy(base), copy.deepcopy(base), copy.deepcopy(base)
    hdl["applications"][0]["resolution"]["hdl_cholesterol"]["assumption_id"] = "A-SYN-02"
    tc["applications"][0]["resolution"]["total_cholesterol"]["assumption_id"] = "A-SYN-02"
    cm["applications"][0]["co_measurement"]["assumption_id"] = "A-SYN-02"
    hashes = [_app_hash(m, rule) for m in (hdl, tc, cm)]
    assert len(set(hashes)) == 3                                   # same assumption set, different links
    assert all(cs.applications_from_mapping(m)[0].assumption_ids == ("A-SYN-01", "A-SYN-02") for m in (hdl, tc, cm))
    swapped = copy.deepcopy(hdl)                                   # swap the HDL and TC links
    swapped["applications"][0]["resolution"]["hdl_cholesterol"]["assumption_id"] = "A-SYN-01"
    swapped["applications"][0]["resolution"]["total_cholesterol"]["assumption_id"] = "A-SYN-02"
    assert _app_hash(swapped, rule) == hashes[1] != hashes[0]


def test_changing_a_link_after_authorisation_invalidates_it():
    rule, app, refs = _setup()
    df = _df(_panel("001", "200", "50", "150", 2))
    extra = {**app.assumptions, "A-SYN-02": cs.Assumption("A-SYN-02", "second", "test", "2026-11-24")}
    for change in ({"resolution_assumptions": {**app.resolution_assumptions, "hdl_cholesterol": "A-SYN-02"},
                    "assumptions": extra},
                   {"co_measurement_assumption": "A-SYN-02", "assumptions": extra}):
        moved = dataclasses.replace(app, **change)   # keeps the authorisation stamped for the original links
        assert cs.application_content_sha256(moved, rule) != app.run_authorization.content_sha256
        with pytest.raises(cs.ConsistencyError, match="must be re-authorised"):
            cs.evaluate(rule, moved, refs, df)
    assert len(cs.evaluate(rule, app, refs, df).coverage) == 1   # the original links still run
