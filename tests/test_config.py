"""Structural validation of source configurations."""

import pytest

from pghd_ready.core.config import ConfigError, SourceConfig, normalize_header

import synthetic_sources as syn
from synthetic_sources import DEVICE, PROFILE


def test_demo_config_parses():
    cfg = SourceConfig.from_mapping(syn.demo_config())
    assert cfg.source == "wearable_demo"
    assert [t.name for t in cfg.tables] == ["device_export", "app_export", "profile"]
    assert "documented_error_code" in cfg.referenced_flags()
    assert cfg.referenced_assumptions() == {"A-DEMO-TZ-01"}


def test_header_normalisation():
    assert normalize_header("﻿  Sensor   B ") == "sensor b"


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda c: c.update(sourc="x"), "unknown keys"),
        (lambda c: c.update(source="Bad-Name"), "lower_snake_case"),
        (lambda c: c["tables"][DEVICE]["observations"][0].update(colour="red"), "unknown keys"),
        (lambda c: c["tables"][DEVICE]["files"].update(path="x.csv"), "exactly one of 'glob' or 'path'"),
        (lambda c: c["tables"][DEVICE]["observations"][1].update(stream_id="patch_a"), "stream_id values must be unique"),
        (lambda c: c["tables"][DEVICE]["observations"][3]["scale"].update(basis="structural", evidence=None),
         "basis must be 'documented' or 'assumption'"),
        (lambda c: c["tables"][DEVICE]["observations"][3]["scale"].pop("evidence"), "requires 'evidence'"),
        (lambda c: c["tables"][DEVICE]["observations"][0].update(
            transforms=[{"name": "x.y", "basis": "assumption"}]), "requires 'assumption_ids'"),
        (lambda c: c["tables"][DEVICE].pop("timestamp"), "require a 'timestamp'"),
        (lambda c: c["tables"][DEVICE]["timestamp"].pop("basis"), "requires a 'basis'"),
        (lambda c: c["tables"][DEVICE]["timestamp"].update(basis="structural", assumption_ids=None),
         "not structural"),
        (lambda c: c["tables"][DEVICE]["participant_id"].update(regex=r"user_\d+"), "exactly one capture group"),
        (lambda c: c["tables"][PROFILE]["attributes"][3]["rules"][0].update(when={"approx": 1}), "exactly one of"),
        (lambda c: c["tables"][DEVICE]["events"][0]["subtype"]["map"].update(other=["lunch"]), "mapped to two"),
        (lambda c: c["tables"][DEVICE]["events"][0]["attributes"]["image_ref"].update(unit="g"), "text attributes"),
    ],
)
def test_invalid_configs_are_rejected(mutate, match):
    cfg = syn.wide_config()
    mutate(cfg)
    with pytest.raises(ConfigError, match=match):
        SourceConfig.from_mapping(cfg)


def test_undefined_assumption_is_rejected(tmp_path):
    from pghd_ready.core.ingest import ingest_source

    cfg = syn.wide_config(plugins=["synthetic_plugin"])
    cfg["tables"][DEVICE]["observations"][0]["transforms"] = [
        {"name": "synthetic.identity", "basis": "assumption", "assumption_ids": ["A-UNDEFINED"]}]
    root = syn.write_wide_source(tmp_path / "raw")
    with pytest.raises(ConfigError, match="undefined assumptions"):
        ingest_source(SourceConfig.from_mapping(cfg), root, syn.concepts())
