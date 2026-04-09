"""Reproducibility: run ID generation, config snapshot, env capture."""
import hashlib
import json
import datetime
from pathlib import Path
from typing import Dict, Any, Optional


def generate_run_id(cfg: dict = None, prefix: str = "") -> str:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if cfg:
        h = hashlib.md5(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()[:6]
        return f"{prefix}{ts}_{h}" if prefix else f"{ts}_{h}"
    return f"{prefix}{ts}" if prefix else ts


def get_env_snapshot() -> Dict[str, Any]:
    env = {}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
    except ImportError:
        pass
    try:
        import numpy as np
        env["numpy"] = np.__version__
    except ImportError:
        pass
    try:
        import mne
        env["mne"] = mne.__version__
    except ImportError:
        pass
    try:
        import sklearn
        env["scikit-learn"] = sklearn.__version__
    except ImportError:
        pass
    return env


def save_run_manifest(
    run_id: str,
    cfg: dict,
    log_dir: str,
    extra: Optional[Dict] = None,
) -> str:
    manifest = {
        "run_id": run_id,
        "timestamp": datetime.datetime.now().isoformat(),
        "config": cfg,
        "env": get_env_snapshot(),
    }
    if extra:
        manifest.update(extra)
    out = Path(log_dir) / "run_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    return str(out)
