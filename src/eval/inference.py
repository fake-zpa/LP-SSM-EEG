"""Inference utilities: run model on a dataset split and save predictions."""
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.eval.metrics_classification import compute_metrics
from src.utils.io import save_json

logger = logging.getLogger(__name__)


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: str = "cuda",
    amp: bool = True,
    output_dir: Optional[str] = None,
    split: str = "test",
) -> Dict:
    """
    Run model inference on a DataLoader.
    Returns: dict with metrics, probabilities, predictions, targets.
    """
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)

    all_logits = []
    all_targets = []

    ctx = torch.cuda.amp.autocast(enabled=(amp and torch.cuda.is_available()))
    with ctx:
        for batch in tqdm(loader, desc=f"Inference [{split}]"):
            x, y = batch
            x = x.to(dev, non_blocking=True)
            out = model(x)
            logits = out["logits"] if isinstance(out, dict) else out
            all_logits.append(logits.float().cpu())
            all_targets.append(y.cpu())

    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)

    metrics = compute_metrics(logits_cat, targets_cat, prefix=split)

    probs = torch.softmax(logits_cat, dim=-1).numpy()
    preds = logits_cat.argmax(dim=-1).numpy()
    targets_np = targets_cat.numpy()

    result = {
        "metrics": metrics,
        "probabilities": probs.tolist(),
        "predictions": preds.tolist(),
        "targets": targets_np.tolist(),
    }

    if output_dir:
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        save_json(metrics, p / f"{split}_metrics.json")
        np.save(p / f"{split}_probs.npy", probs)
        np.save(p / f"{split}_preds.npy", preds)
        np.save(p / f"{split}_targets.npy", targets_np)
        logger.info(f"Saved inference outputs to {output_dir}")

    return result
