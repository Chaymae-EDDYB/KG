from __future__ import annotations
import yaml
from pathlib import Path
from typing import Any, Dict, Optional, Union

def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result

def load_config(override_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    default = Path("configs/default.yaml")
    if not default.exists():
        raise FileNotFoundError(
            f"Config not found at {default}. Run commands from the project root."
        )
    with open(default) as f:
        cfg = yaml.safe_load(f) or {}
    if override_path:
        with open(override_path) as f:
            cfg = _deep_merge(cfg, yaml.safe_load(f) or {})
    return cfg
