import pytest

from pghd_ready.core.annotations import (
    AnnotationError,
    AnnotationRegistry,
    Assumption,
    FlagAction,
    FlagDefinition,
    IssueCategory,
    summarize_flags,
    validate_flags,
)

from conftest import SYNTHETIC_ANNOTATIONS, make_attributes, make_observations


def test_registry_loads_all_four_categories(annotation_registry):
    cats = {f.category for f in annotation_registry.flags.values()}
    assert cats == set(IssueCategory)


@pytest.mark.parametrize(
    "category",
    [IssueCategory.SUSPECTED_ANOMALY, IssueCategory.SEMANTIC_AMBIGUITY, IssueCategory.ASSUMPTION_BASED],
)
def test_only_documented_issues_may_null_values(category):
    kwargs = {"assumption_id": "A-X"} if category is IssueCategory.ASSUMPTION_BASED else {}
    with pytest.raises(AnnotationError, match="not permitted"):
        FlagDefinition("x", category, "desc", action=FlagAction.SET_NULL, **kwargs)


def test_documented_issue_requires_evidence():
    with pytest.raises(AnnotationError, match="evidence"):
        FlagDefinition("x", IssueCategory.DOCUMENTED_ISSUE, "desc", action=FlagAction.SET_NULL)


def test_assumption_flag_requires_registered_assumption():
    with pytest.raises(AnnotationError, match="assumption_id"):
        FlagDefinition("x", IssueCategory.ASSUMPTION_BASED, "desc")
    reg = AnnotationRegistry()
    with pytest.raises(AnnotationError, match="unknown assumption"):
        reg.add_flag(FlagDefinition("x", IssueCategory.ASSUMPTION_BASED, "desc", assumption_id="A-MISSING"))


def test_non_assumption_flag_cannot_reference_assumption():
    with pytest.raises(AnnotationError, match="only assumption-based"):
        FlagDefinition("x", IssueCategory.SUSPECTED_ANOMALY, "desc", assumption_id="A-SYN-01")


def test_invalid_codes_and_ids_rejected():
    with pytest.raises(AnnotationError):
        FlagDefinition("Bad-Code", IssueCategory.SUSPECTED_ANOMALY, "desc")
    with pytest.raises(AnnotationError):
        Assumption("lowercase", "s", "r")


def test_unknown_category_in_config_rejected():
    with pytest.raises(AnnotationError, match="category"):
        AnnotationRegistry.from_mapping({"flags": {"x": {"category": "probably_fine", "description": "d"}}})


def test_merge_detects_conflicts(annotation_registry):
    other = AnnotationRegistry.from_mapping(
        {"flags": {"suspected_spike": {"category": "semantic_ambiguity", "description": "different"}}}
    )
    with pytest.raises(AnnotationError, match="Conflicting"):
        annotation_registry.merge(other)
    # identical definitions merge cleanly
    same = AnnotationRegistry.from_mapping(SYNTHETIC_ANNOTATIONS)
    assert annotation_registry.merge(same).flags == annotation_registry.flags


def test_validate_and_summarize_flags(annotation_registry):
    obs = make_observations()
    validate_flags(obs, annotation_registry)
    summary = summarize_flags(make_attributes(), annotation_registry)
    assert summary.to_dict("records") == [
        {"category": "documented_issue", "code": "documented_error_code", "n_records": 1}
    ]
    obs.at[0, "quality_flags"] = ["never_registered"]
    with pytest.raises(AnnotationError, match="never_registered"):
        validate_flags(obs, annotation_registry)


def test_registry_snapshot_is_serialisable(annotation_registry):
    snap = annotation_registry.to_dict()
    assert snap["flags"]["documented_error_code"]["category"] == "documented_issue"
    assert snap["assumptions"]["A-SYN-01"]["statement"]
