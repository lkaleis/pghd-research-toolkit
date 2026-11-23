"""Synthetic test sources.

The main scenario is the persisted, fully fictional demo dataset in
``data/synthetic/wearable_demo`` (raw files + source YAML). Tests copy it into a
temporary directory so they can modify files without touching the original.

Small builders for long-format/timezone edge cases remain here because they are
easier to read inline.
"""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import yaml

from pghd_ready.core.concepts import ConceptRegistry

REPO_ROOT = Path(__file__).resolve().parents[1]
DEMO_DIR = REPO_ROOT / "data" / "synthetic" / "wearable_demo"
DEMO_CONFIG = DEMO_DIR / "config" / "source.yaml"
SHARED_CONCEPTS = REPO_ROOT / "configs" / "concepts" / "shared.yaml"
CARDIO_CONCEPTS = REPO_ROOT / "configs" / "conditions" / "cardiometabolic.yaml"
CONCEPT_FILES = [SHARED_CONCEPTS, CARDIO_CONCEPTS]

# table positions in the demo configuration
DEVICE, APP, PROFILE = 0, 1, 2
PID = "wearable_demo:{:03d}"

# extra annotations used only by transform tests (not part of the demo dataset)
TEST_ANNOTATIONS = {
    "assumptions": {
        "A-SYN-01": {"statement": "Every second reading is native.", "rationale": "test"},
        "A-SYN-02": {"statement": "An unrelated assumption.", "rationale": "test"},
    },
    "flags": {
        "lattice_reading": {"category": "assumption_based", "description": "Kept as native reading.",
                            "assumption_id": "A-SYN-01"},
        "inferred_native": {"category": "assumption_based", "assumption_id": "A-SYN-01",
                            "description": "Classified as a native reading (inferred, not verified)."},
        "inferred_interpolated": {"category": "assumption_based", "assumption_id": "A-SYN-01",
                                  "description": "Classified as an interpolated value (inferred, not verified)."},
        "origin_undetermined": {"category": "assumption_based", "assumption_id": "A-SYN-01",
                                "description": "Classification attempted; evidence insufficient, origin left unknown."},
        "other_assumption_flag": {"category": "assumption_based", "description": "Unrelated assumption.",
                                  "assumption_id": "A-SYN-02"},
    },
}


def concepts() -> ConceptRegistry:
    return ConceptRegistry.from_yaml_files(CONCEPT_FILES)


def write_wide_source(root: Path) -> Path:
    """Copy the persisted demo raw files to ``root`` (a temporary directory)."""
    shutil.copytree(DEMO_DIR / "raw", root)
    return root


def demo_config() -> dict:
    return yaml.safe_load(DEMO_CONFIG.read_text(encoding="utf-8"))


def wide_config(with_test_annotations: bool = True, **overrides) -> dict:
    cfg = copy.deepcopy(demo_config())
    if with_test_annotations:
        cfg["annotations"]["assumptions"].update(copy.deepcopy(TEST_ANNOTATIONS["assumptions"]))
        cfg["annotations"]["flags"].update(copy.deepcopy(TEST_ANNOTATIONS["flags"]))
    cfg.update(overrides)
    return cfg


def write_config(path: Path, cfg: dict) -> Path:
    path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return path


# --------------------------------------------------------------------------- long "app export" edge cases
def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_long_source(root: Path, with_offsets: bool = True) -> Path:
    if with_offsets:
        rows = [
            "2021-03-28T00:30:00+00:00,u1,glucose,5.5",
            "2021-03-28T03:30:00+02:00,u1,glucose,6.0",  # same zone, after DST change
            "2021-03-28T01:45:00Z,u1,steps,120",
            "2021-03-28T02:00:00+00:00,u2,glucose,7.1",
        ]
    else:
        rows = [
            "2021-03-28 01:30:00,u1,glucose,5.5",
            "2021-03-28 02:30:00,u1,glucose,6.0",  # does not exist in Europe/Berlin (DST gap)
            "2021-03-28 03:30:00,u1,glucose,6.2",
            "2021-03-28 03:45:00,u1,steps,120",
        ]
    _write(root / "export.csv", "recorded_at,user,type,value\n" + "\n".join(rows) + "\n")
    return root


def long_config(timestamp: dict) -> dict:
    return {
        "source": "synthetic_app",
        "version": "0.1",
        "temporal": {"output_timezone": "UTC"},
        "annotations": {"assumptions": {"A-TZ": {"statement": "Times are Berlin local time.", "rationale": "test"}}},
        "tables": [{
            "name": "readings",
            "files": {"path": "export.csv"},
            "participant_id": {"from": "column", "column": ["user"]},
            "timestamp": timestamp,
            "observations": [
                {"stream_id": "app_glucose", "concept": "glucose_interstitial", "column": ["value"], "unit": "mmol/L",
                 "where": {"column": ["type"], "equals": "glucose"}},
            ],
        }],
    }
