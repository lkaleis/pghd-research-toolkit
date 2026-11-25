"""Command-line interface.

    pghd-ready ingest --source configs/sources/<source>.yaml \
                      --concepts configs/concepts/shared.yaml configs/conditions/<condition>.yaml \
                      --data-root data/raw/<source> --out data/processed/<source>

    pghd-ready profile --m1 data/processed/<source> --out data/processed/<source>/quality \
                       [--expected-interval STREAM=SECONDS ...] [--gap-factor 1.5]

    pghd-ready prepare --task configs/tasks/<task>.yaml --m1 data/processed/<source> \
                       --out data/processed/<source>/tasks/<task>

Exit codes: 0 = success, 0 = completed with warnings (see the report), 1 = failed.
An ``ingestion_report.md`` is written to ``--out`` in every case.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .core.ingest import ingest_and_write


def _ingest(args: argparse.Namespace) -> int:
    report = args.out / "ingestion_report.md"
    try:
        outcome = ingest_and_write(args.source, args.concepts, args.data_root, args.out, base_dir=args.base_dir)
    except Exception as exc:  # the report has already been written
        print(f"Ingestion FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        print(f"Report: {report}", file=sys.stderr)
        return 1
    rows = outcome.result.summary()["rows"]
    label = {"success": "SUCCESS", "warnings": "COMPLETED WITH WARNINGS"}[outcome.status]
    print(f"Ingestion {label}: " + ", ".join(f"{k}={v}" for k, v in rows.items()))
    print(f"Report: {report}")
    return 0


def _interval(text: str) -> tuple[str, float]:
    stream, sep, seconds = text.partition("=")
    if not sep or not stream:
        raise argparse.ArgumentTypeError(f"expected STREAM=SECONDS, got {text!r}")
    return stream, float(seconds)


def _profile(args: argparse.Namespace) -> int:
    from .core.profiling import profile_m1_run
    try:
        profile = profile_m1_run(args.m1, args.out, args.base_dir, dict(args.expected_interval or []), args.gap_factor)
    except Exception as exc:
        print(f"Profiling FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Profiling SUCCESS: {len(profile.dataset_summary)} streams, "
          f"{len(profile.stream_profile)} participant-stream profiles")
    print(f"Report: {args.out / 'profile_report.md'}")
    return 0


def _prepare(args: argparse.Namespace) -> int:
    import importlib

    import yaml
    try:
        with open(args.task, encoding="utf-8") as fh:
            module = importlib.import_module((yaml.safe_load(fh) or {})["module"])
        result = module.run(args.task, args.m1, args.out, args.base_dir)
    except Exception as exc:
        print(f"Task preparation FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(f"Task preparation SUCCESS: {len(result.ledger)} candidates, {len(result.analysis_table)} analysis rows")
    print(f"Report: {args.out / 'task_report.md'}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pghd-ready")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ingest", help="Ingest a tabular source described by a YAML configuration.")
    p.add_argument("--source", required=True, type=Path, help="source configuration YAML")
    p.add_argument("--concepts", required=True, type=Path, nargs="+",
                   help="one or more concept registry YAML files (e.g. shared + condition); merged, conflicts fail")
    p.add_argument("--data-root", required=True, type=Path, help="directory containing the source files")
    p.add_argument("--out", required=True, type=Path, help="output directory (should be git-ignored)")
    p.add_argument("--base-dir", type=Path, default=Path.cwd(), help="base for relative paths in manifest and report")
    p.set_defaults(func=_ingest)
    p = sub.add_parser("profile", help="Descriptive data-quality profile of a finished M1 run (M2b).")
    p.add_argument("--m1", required=True, type=Path, help="M1 output directory (e.g. data/processed/<source>)")
    p.add_argument("--out", required=True, type=Path, help="output directory (should be git-ignored)")
    p.add_argument("--expected-interval", type=_interval, action="append", metavar="STREAM=SECONDS",
                   help="explicitly configured expected interval for a stream without a documented one")
    p.add_argument("--gap-factor", type=float, default=1.5,
                   help="a gap is an interval longer than this factor x the expected interval (default 1.5)")
    p.add_argument("--base-dir", type=Path, default=Path.cwd(), help="base for relative paths in the manifest")
    p.set_defaults(func=_profile)
    p = sub.add_parser("prepare", help="Analytical preparation for a task (M3): windows, eligibility, analysis table.")
    p.add_argument("--task", required=True, type=Path, help="task configuration YAML (configs/tasks/...)")
    p.add_argument("--m1", required=True, type=Path, help="M1 output directory")
    p.add_argument("--out", required=True, type=Path, help="output directory (should be git-ignored)")
    p.add_argument("--base-dir", type=Path, default=Path.cwd(), help="base for relative paths in the manifest")
    p.set_defaults(func=_prepare)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
