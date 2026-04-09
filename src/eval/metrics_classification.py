"""Classification metrics for EEG experiments."""
import torch
import numpy as np
from typing import Dict, Optional
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
)


def compute_metrics(
    logits: torch.Tensor,
    targets: torch.Tensor,
    threshold: Optional[float] = None,
    prefix: str = "val",
) -> Dict[str, float]:
    """
    Compute classification metrics from logits and integer targets.

    logits:  [N, n_classes]
    targets: [N]
    """
    probs = torch.softmax(logits.float(), dim=-1).numpy()
    y_true = targets.numpy()

    n_classes = probs.shape[1]
    pos_probs = probs[:, 1] if n_classes == 2 else probs

    results = {}

    try:
        if n_classes == 2:
            results[f"{prefix}_auroc"] = float(roc_auc_score(y_true, pos_probs))
            results[f"{prefix}_auprc"] = float(average_precision_score(y_true, pos_probs))
        else:
            results[f"{prefix}_auroc"] = float(roc_auc_score(y_true, pos_probs, multi_class="ovr"))
    except Exception:
        results[f"{prefix}_auroc"] = 0.0
        results[f"{prefix}_auprc"] = 0.0

    if threshold is None:
        thresholds = np.linspace(0.1, 0.9, 17)
        best_f1 = 0.0
        best_thresh = 0.5
        for t in thresholds:
            preds_t = (pos_probs >= t).astype(int)
            f = f1_score(y_true, preds_t, average="binary" if n_classes == 2 else "macro", zero_division=0)
            if f > best_f1:
                best_f1 = f
                best_thresh = t
        threshold = best_thresh

    preds = (pos_probs >= threshold).astype(int) if n_classes == 2 else probs.argmax(axis=1)

    results[f"{prefix}_threshold"] = float(threshold)
    results[f"{prefix}_f1"] = float(f1_score(y_true, preds, average="binary" if n_classes == 2 else "macro", zero_division=0))
    results[f"{prefix}_balanced_acc"] = float(balanced_accuracy_score(y_true, preds))
    results[f"{prefix}_acc"] = float((preds == y_true).mean())

    if n_classes == 2:
        cm = confusion_matrix(y_true, preds, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            results[f"{prefix}_sensitivity"] = float(tp / max(tp + fn, 1))
            results[f"{prefix}_specificity"] = float(tn / max(tn + fp, 1))
        else:
            results[f"{prefix}_sensitivity"] = 0.0
            results[f"{prefix}_specificity"] = 0.0

    return results
