from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class Config(dict):
    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        if isinstance(value, dict) and not isinstance(value, Config):
            value = Config(value)
            self[name] = value
        return value


def _to_config(value: Any) -> Any:
    if isinstance(value, dict):
        return Config({key: _to_config(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_to_config(item) for item in value]
    return value


def load_config(path: str | Path) -> Config:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        data = {}
    return _to_config(data)
