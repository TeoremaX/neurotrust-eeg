from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    """Recursively update a dictionary without mutating the original."""

    result = dict(base)

    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value

    return result


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file, optionally inheriting from another config."""

    path = Path(path)

    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if config is None:
        return {}

    parent_path = config.get("inherits")

    if parent_path:
        parent_config = load_config(parent_path)
        child_config = {key: value for key, value in config.items() if key != "inherits"}
        return deep_update(parent_config, child_config)

    return config