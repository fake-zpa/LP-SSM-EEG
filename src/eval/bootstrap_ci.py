"""Bootstrap confidence interval estimation for evaluation metrics."""
import numpy as np
from typing import Callable, Dict, Tuple


def bootstrap_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable,
    n_iterations: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap CI for a scalar metric.

    Returns: (mean, lower, upper)
    """
    rng = np.random.RandomState(seed)
    n = len(y_true)
    scores = []
    for _ in range(n_iterations):
        idx = rng.choice(n, size=n, replace=True)
        try:
            s = metric_fn(y_true[idx], y_score[idx])
            scores.append(s)
        except Exception:
            pass

    scores = np.array(scores)
    alpha = 1 - confidence
    lower = float(np.percentile(scores, 100 * alpha / 2))
    upper = float(np.percentile(scores, 100 * (1 - alpha / 2)))
    return float(scores.mean()), lower, upper


def compute_all_ci(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_iterations: int = 1000,
    seed: int = 42,
) -> Dict[str, Dict]:
    """Compute bootstrap CIs for AUROC and AUPRC."""
    from sklearn.metrics import roc_auc_score, average_precision_score
    results = {}
    for name, fn in [("auroc", roc_auc_score), ("auprc", average_precision_score)]:
        mean, lo, hi = bootstrap_ci(y_true, y_score, fn, n_iterations=n_iterations, seed=seed)
        results[name] = {"mean": mean, "ci_lower": lo, "ci_upper": hi}
    return results
