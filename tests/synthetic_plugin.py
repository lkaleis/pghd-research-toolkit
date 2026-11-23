"""Synthetic transform plugin used only by the test suite (some transforms misbehave on purpose)."""

import pandas as pd

from pghd_ready.core.transforms import register_transform


@register_transform("synthetic.keep_every_other")
def keep_every_other(df, params, ctx):
    """Keep every n-th row and flag kept rows as assumption-based (mimics native-sample selection)."""
    out = df.iloc[:: int(params.get("step", 2))].reset_index(drop=True)
    out["quality_flags"] = [list(f) + ["lattice_reading"] for f in out["quality_flags"]]
    return out


@register_transform("synthetic.change_with_flag")
def change_with_flag(df, params, ctx):
    """Change every value and add the flag named in params (legitimate only if that flag explains the change)."""
    df = df.copy()
    df["value"] = df["value"] + 1.0
    df["quality_flags"] = [list(f) + [params["flag"]] for f in df["quality_flags"]]
    return df


@register_transform("synthetic.silent_change")
def silent_change(df, params, ctx):
    df = df.copy()
    df["value"] = df["value"] + 1.0
    return df


@register_transform("synthetic.tamper_raw")
def tamper_raw(df, params, ctx):
    df = df.copy()
    df["raw_value"] = "edited"
    df["quality_flags"] = [list(f) + ["lattice_reading"] for f in df["quality_flags"]]
    return df


@register_transform("synthetic.add_row")
def add_row(df, params, ctx):
    """Append a copy of the first row one minute later, with the flags given in params."""
    new = df.iloc[[0]].copy()
    new["timestamp"] = new["timestamp"] + pd.Timedelta(minutes=1, seconds=30)
    new["quality_flags"] = [list(params.get("flags", []))]
    return pd.concat([df, new], ignore_index=True)


@register_transform("synthetic.classify_origin")
def classify_origin(df, params, ctx):
    """Toy sampling-origin classifier used to test the representation (not a real method).

    Readings on minutes divisible by ``lattice_min`` are classified native, others interpolated, but only
    when the group has at least ``min_readings`` readings; otherwise the evidence is deemed insufficient
    and every reading stays unknown. Values and timestamps are never changed.
    """
    df = df.copy()
    k, min_n = int(params["lattice_min"]), int(params["min_readings"])
    confident = len(df) >= min_n
    ctx.record("method", "toy_minute_lattice_v0")
    ctx.record("lattice_min", k)
    ctx.record("evidence_readings", len(df))
    ctx.record("confidence", 0.9 if confident else 0.0)
    if not confident:
        # contract: only annotate readings whose origin is still unknown; leave declared readings untouched
        unknown = df["is_interpolated"].isna().to_numpy()
        df["quality_flags"] = [list(f) + (["origin_undetermined"] if u else [])
                               for f, u in zip(df["quality_flags"], unknown)]
        return df
    on_lattice = (df["timestamp"].dt.minute % k == 0).to_numpy()
    df["is_interpolated"] = pd.array(~on_lattice, dtype="boolean")
    df["quality_flags"] = [list(f) + (["inferred_native"] if nat else ["inferred_interpolated"])
                           for f, nat in zip(df["quality_flags"], on_lattice)]
    return df


@register_transform("synthetic.classify_without_flag")
def classify_without_flag(df, params, ctx):
    """Misbehaving classifier: sets is_interpolated without an explanatory flag."""
    df = df.copy()
    df["is_interpolated"] = pd.array([True] * len(df), dtype="boolean")
    return df


@register_transform("synthetic.identity")
def identity(df, params, ctx):
    return df
