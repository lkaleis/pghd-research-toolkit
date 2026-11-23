"""Provenance: run manifests and transformation logs.

* :class:`RunManifest` records what went in (file hashes), how (config hashes,
  toolkit version, git commit, annotation registry snapshot) and what came out
  (files, row counts).
* :class:`TransformLog` records each transformation step with its parameters,
  row counts, flags raised and its *basis*:

  - ``structural``: reshaping/renaming with no change of meaning;
  - ``documented``: justified by the source's documentation (cite evidence);
  - ``assumption``: relies on registered assumptions (ids required).

Paths are stored relative to a base directory so manifests contain no
machine-specific absolute paths.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping

from .. import __version__
from .annotations import AnnotationError, AnnotationRegistry

Clock = Callable[[], datetime]


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)


def config_hash(obj: Any) -> str:
    """Stable hash of a configuration object (key order does not matter)."""
    return hashlib.sha256(canonical_json(obj).encode("utf-8")).hexdigest()


def git_commit(cwd: str | Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=10, check=True
        )
        commit = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=cwd, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


class TransformBasis(str, Enum):
    STRUCTURAL = "structural"
    DOCUMENTED = "documented"
    ASSUMPTION = "assumption"


@dataclass
class TransformRecord:
    step: str
    function: str
    basis: TransformBasis
    rows_in: int
    rows_out: int
    params: dict[str, Any] = field(default_factory=dict)
    assumption_ids: list[str] = field(default_factory=list)
    evidence: str | None = None
    flags_raised: dict[str, int] = field(default_factory=dict)
    notes: str | None = None
    recorded_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["basis"] = self.basis.value
        return d


class TransformLog:
    """Append-only log of transformation steps, validated against the annotation registry."""

    def __init__(self, registry: AnnotationRegistry | None = None, clock: Clock = _utc_now):
        self.registry = registry
        self.clock = clock
        self.records: list[TransformRecord] = []

    def record(
        self,
        step: str,
        function: str,
        basis: TransformBasis | str,
        rows_in: int,
        rows_out: int,
        params: Mapping[str, Any] | None = None,
        assumption_ids: list[str] | None = None,
        evidence: str | None = None,
        flags_raised: Mapping[str, int] | None = None,
        notes: str | None = None,
    ) -> TransformRecord:
        basis = TransformBasis(basis)
        assumption_ids = list(assumption_ids or [])
        if basis is TransformBasis.ASSUMPTION and not assumption_ids:
            raise AnnotationError(f"Step {step!r}: assumption-based transforms must list assumption_ids.")
        if basis is not TransformBasis.ASSUMPTION and assumption_ids:
            raise AnnotationError(f"Step {step!r}: assumption_ids given but basis is {basis.value!r}.")
        if basis is TransformBasis.DOCUMENTED and not evidence:
            raise AnnotationError(f"Step {step!r}: documented transforms must cite evidence.")
        if self.registry is not None:
            missing = [a for a in assumption_ids if a not in self.registry.assumptions]
            if missing:
                raise AnnotationError(f"Step {step!r}: unknown assumptions {missing}.")
            unknown = self.registry.unknown_codes((flags_raised or {}).keys())
            if unknown:
                raise AnnotationError(f"Step {step!r}: unregistered flags {sorted(unknown)}.")
        rec = TransformRecord(
            step=step,
            function=function,
            basis=basis,
            rows_in=int(rows_in),
            rows_out=int(rows_out),
            params=dict(params or {}),
            assumption_ids=assumption_ids,
            evidence=evidence,
            flags_raised={k: int(v) for k, v in (flags_raised or {}).items()},
            notes=notes,
            recorded_at=self.clock().isoformat(),
        )
        self.records.append(rec)
        return rec

    def write_jsonl(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            for rec in self.records:
                fh.write(canonical_json(rec.to_dict()) + "\n")


class RunManifest:
    """Builder for a run manifest; paths are recorded relative to ``base_dir``."""

    def __init__(self, run_name: str, base_dir: str | Path, clock: Clock = _utc_now, record_git: bool = True):
        self.base_dir = Path(base_dir).resolve()
        self.data: dict[str, Any] = {
            "run_name": run_name,
            "created_at": clock().isoformat(),
            "toolkit_version": __version__,
            "git_commit": git_commit(self.base_dir) if record_git else None,
            "configs": {},
            "source": {},
            "inputs": [],
            "outputs": [],
            "annotations": {},
        }

    def _rel(self, path: str | Path) -> str:
        p = Path(path).resolve()
        try:
            return p.relative_to(self.base_dir).as_posix()
        except ValueError as exc:
            raise ValueError(f"{p} is outside the manifest base directory {self.base_dir}.") from exc

    def add_config(self, name: str, config: Any) -> str:
        h = config_hash(config)
        self.data["configs"][name] = {"sha256": h}
        return h

    def set_source(self, metadata: Mapping[str, Any]) -> None:
        self.data["source"] = dict(metadata)

    def set_annotations(self, registry: AnnotationRegistry) -> None:
        self.data["annotations"] = registry.to_dict()

    def add_input(self, path: str | Path) -> dict[str, Any]:
        p = Path(path)
        entry = {"path": self._rel(p), "sha256": sha256_file(p), "bytes": p.stat().st_size}
        self.data["inputs"].append(entry)
        return entry

    def add_output(self, path: str | Path, table: str | None = None, rows: int | None = None) -> dict[str, Any]:
        p = Path(path)
        entry = {"path": self._rel(p), "table": table, "rows": rows, "sha256": sha256_file(p)}
        self.data["outputs"].append(entry)
        return entry

    def to_dict(self) -> dict[str, Any]:
        d = json.loads(canonical_json(self.data))
        d["inputs"] = sorted(d["inputs"], key=lambda e: e["path"])
        d["outputs"] = sorted(d["outputs"], key=lambda e: e["path"])
        return d

    def write(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh, indent=2, sort_keys=True, ensure_ascii=False)
            fh.write("\n")
