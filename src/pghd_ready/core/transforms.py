"""Registry for named transforms.

Transforms are the escape hatch for logic that configuration cannot express.
They are plain functions registered under a dotted name by adapter modules
(declared as ``plugins`` in a source config) and invoked from configuration.

Signature::

    fn(df: pandas.DataFrame, params: dict, context: TransformContext) -> pandas.DataFrame

``df`` holds the observation rows of one participant x stream in the common
representation; the function must return rows in the same representation. It
may drop rows or add flags but must never silently alter values. The engine
validates the result and records rows in/out, flags raised and the configured
basis in the transformation log.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd


class TransformError(RuntimeError):
    pass


@dataclass(frozen=True)
class TransformContext:
    source: str
    participant_id: str
    stream_id: str
    concept: str


TransformFn = Callable[[pd.DataFrame, dict, TransformContext], pd.DataFrame]

_REGISTRY: dict[str, TransformFn] = {}


def register_transform(name: str) -> Callable[[TransformFn], TransformFn]:
    if "." not in name:
        raise TransformError(f"Transform names must be namespaced, e.g. '<adapter>.{name}'.")

    def decorator(fn: TransformFn) -> TransformFn:
        existing = _REGISTRY.get(name)
        if existing is not None and existing is not fn:
            raise TransformError(f"Transform {name!r} is already registered.")
        _REGISTRY[name] = fn
        return fn

    return decorator


def get_transform(name: str) -> TransformFn:
    try:
        return _REGISTRY[name]
    except KeyError as exc:
        raise TransformError(f"Unknown transform {name!r}; is its plugin module listed in 'plugins'?") from exc


def load_plugins(modules: tuple[str, ...] | list[str]) -> None:
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ImportError as exc:
            raise TransformError(f"Cannot import plugin module {mod!r}: {exc}") from exc


def registered_transforms() -> list[str]:
    return sorted(_REGISTRY)


def resolve_callable(ref: str) -> Any:
    """Resolve 'package.module:attribute' to an object."""
    if ":" not in ref:
        raise TransformError(f"Expected 'module:attribute', got {ref!r}.")
    mod, attr = ref.split(":", 1)
    try:
        return getattr(importlib.import_module(mod), attr)
    except (ImportError, AttributeError) as exc:
        raise TransformError(f"Cannot resolve {ref!r}: {exc}") from exc
