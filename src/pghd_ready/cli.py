"""Command-line interface.

    pghd-ready ingest --source configs/sources/<source>.yaml \
                      --concepts configs/conditions/<condition>.yaml \
                      --data-root data/raw/<source> --out data/processed/<source>

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="pghd-ready")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("ingest", help="Ingest a tabular source described by a YAML configuration.")
    p.add_argument("--source", required=True, type=Path, help="source configuration YAML")
    p.add_argument("--concepts", required=True, type=Path, help="condition concept registry YAML")
    p.add_argument("--data-root", required=True, type=Path, help="directory containing the source files")
    p.add_argument("--out", required=True, type=Path, help="output directory (should be git-ignored)")
    p.add_argument("--base-dir", type=Path, default=Path.cwd(), help="base for relative paths in manifest and report")
    p.set_defaults(func=_ingest)
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
