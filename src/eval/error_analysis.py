"""Error analysis: find and summarize misclassified examples."""
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from src.utils.io import save_json


def analyze_errors(
    probs: np.ndarray,
    preds: np.ndarray,
    targets: np.ndarray,
    threshold: float = 0.5,
    output_dir: Optional[str] = None,
) -> Dict:
    """
    Identify false positives, false negatives, and confidence distributions.

    probs:   [N, 2] probability array
    preds:   [N] predicted class
    targets: [N] true class
    """
    pos_probs = probs[:, 1]
    tp_mask = (preds == 1) & (targets == 1)
    fp_mask = (preds == 1) & (targets == 0)
    fn_mask = (preds == 0) & (targets == 1)
    tn_mask = (preds == 0) & (targets == 0)

    result = {
        "n_tp": int(tp_mask.sum()),
        "n_fp": int(fp_mask.sum()),
        "n_fn": int(fn_mask.sum()),
        "n_tn": int(tn_mask.sum()),
        "fp_mean_prob": float(pos_probs[fp_mask].mean()) if fp_mask.any() else None,
        "fn_mean_prob": float(pos_probs[fn_mask].mean()) if fn_mask.any() else None,
        "tp_mean_prob": float(pos_probs[tp_mask].mean()) if tp_mask.any() else None,
        "high_confidence_fp_indices": np.where(fp_mask & (pos_probs > 0.8))[0].tolist()[:20],
        "low_confidence_fn_indices": np.where(fn_mask & (pos_probs < 0.3))[0].tolist()[:20],
    }

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        save_json(result, Path(output_dir) / "error_analysis.json")

    return result
