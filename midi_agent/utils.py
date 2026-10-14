"""Shared helpers."""

from __future__ import annotations

from typing import Any


def make_json_safe(value: Any) -> Any:
    """Convert numpy scalars and nested structures into JSON-serializable values."""
    try:
        import numpy as np
    except ImportError:
        np = None

    if np is not None and isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [make_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [make_json_safe(item) for item in value]
    return value
