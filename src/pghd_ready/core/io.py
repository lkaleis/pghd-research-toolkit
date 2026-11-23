"""Validated Parquet I/O for common-representation tables.

Tables are validated before writing and after reading. Schema name/version and
temporal policy are stored in the Parquet key-value metadata so that a file
identifies what it contains.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pandera.pandas as pa
import pyarrow as pa_arrow
import pyarrow.parquet as pq

_META_KEY = b"pghd_ready"
_LIST_COLUMNS = ("quality_flags",)


def write_table(df: pd.DataFrame, schema: pa.DataFrameSchema, path: str | Path) -> pd.DataFrame:
    """Validate ``df`` against ``schema`` and write it to Parquet. Returns the validated frame."""
    validated = schema.validate(df, lazy=True)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa_arrow.Table.from_pandas(validated, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[_META_KEY] = json.dumps(schema.metadata or {}, sort_keys=True).encode("utf-8")
    pq.write_table(table.replace_schema_metadata(meta), path)
    return validated


def read_metadata(path: str | Path) -> dict:
    meta = pq.read_schema(path).metadata or {}
    raw = meta.get(_META_KEY)
    return json.loads(raw) if raw else {}


def read_table(path: str | Path, schema: pa.DataFrameSchema) -> pd.DataFrame:
    """Read a Parquet table, restore Python list columns and nanosecond timestamps, and validate."""
    stored = read_metadata(path)
    expected = (schema.metadata or {}).get("pghd_table")
    if stored and expected and stored.get("pghd_table") != expected:
        raise ValueError(f"{path} contains table {stored.get('pghd_table')!r}, expected {expected!r}.")
    df = pq.read_table(path).to_pandas()
    for col in _LIST_COLUMNS:
        if col in df.columns:
            df[col] = df[col].map(lambda v: [] if v is None else [str(x) for x in v])
    for col, dtype in df.dtypes.items():
        if isinstance(dtype, pd.DatetimeTZDtype):
            df[col] = df[col].dt.as_unit("ns")
        elif pd.api.types.is_datetime64_dtype(dtype):
            df[col] = df[col].astype("datetime64[ns]")
    return schema.validate(df, lazy=True)
