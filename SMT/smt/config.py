from __future__ import annotations

import json
from pathlib import Path


def load_config(path: str):
    config_path=Path(path).resolve()
    data=json.loads(config_path.read_text(encoding="utf-8"))
    data["_config_path"]=str(config_path)
    data["_config_dir"]=str(config_path.parent)
    return data


def resolve_path(config, value: str) -> Path:
    p=Path(value)
    if p.is_absolute():
        return p
    return (Path(config["_config_dir"]) / p).resolve()


def ensure_parent(path: Path):
    path.parent.mkdir(parents=True,exist_ok=True)
