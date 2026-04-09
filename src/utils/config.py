"""Config loading and merging utilities (YAML-based)."""
import json
import copy
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base."""
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = deep_merge(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def load_config(config_path: str, defaults_path: Optional[str] = None) -> Dict[str, Any]:
    """Load config YAML, optionally merged on top of defaults."""
    cfg = {}
    if defaults_path and Path(defaults_path).exists():
        cfg = load_yaml(defaults_path)
    override = load_yaml(config_path)
    return deep_merge(cfg, override)


def save_config_snapshot(cfg: dict, output_path: str) -> None:
    """Save config dict to YAML snapshot (for reproducibility)."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)


def cfg_get(cfg: dict, key_path: str, default: Any = None) -> Any:
    """Dot-notation access into nested dict. E.g. cfg_get(cfg, 'training.lr')."""
    parts = key_path.split(".")
    node = cfg
    for p in parts:
        if not isinstance(node, dict) or p not in node:
            return default
        node = node[p]
    return node
