"""Guard: the shared core must not contain dataset- or condition-specific knowledge."""

import re

import pytest

from conftest import REPO_ROOT

CORE_DIR = REPO_ROOT / "src" / "pghd_ready" / "core"
FORBIDDEN = re.compile(r"cgmacros|libre|dexcom|fitbit|glucose|insulin|cholesterol|meal", re.IGNORECASE)


@pytest.mark.parametrize("path", sorted(CORE_DIR.glob("*.py")), ids=lambda p: p.name)
def test_core_module_has_no_specific_knowledge(path):
    hits = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if FORBIDDEN.search(line):
            hits.append(f"{path.name}:{lineno}: {line.strip()}")
    assert not hits, "Dataset/condition-specific terms in core:\n" + "\n".join(hits)


def test_core_does_not_import_conditions_or_adapters():
    for path in CORE_DIR.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "conditions" not in re.findall(r"^\s*(?:from|import)\s+\S+", text, re.MULTILINE).__str__()
        assert "adapters" not in re.findall(r"^\s*(?:from|import)\s+\S+", text, re.MULTILINE).__str__()
