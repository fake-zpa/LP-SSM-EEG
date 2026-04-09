"""Learning rate scheduler factory."""
import math
import torch
from typing import Any, Dict


def build_scheduler(optimizer, cfg: Dict[str, Any], steps_per_epoch: int = 1, max_epochs: int = 100):
    name = cfg.get("name", "cosine").lower()

    if name == "cosine":
        T_max = cfg.get("T_max", max_epochs)
        eta_min = cfg.get("eta_min", 1e-6)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    elif name == "cosine_warmup":
        warmup_steps = cfg.get("warmup_steps", 500)
        T_max = steps_per_epoch * max_epochs
        eta_min = cfg.get("eta_min", 1e-6)

        def lr_lambda(step):
            if step < warmup_steps:
                return step / max(warmup_steps, 1)
            progress = (step - warmup_steps) / max(T_max - warmup_steps, 1)
            return eta_min + 0.5 * (1.0 - eta_min) * (1 + math.cos(math.pi * progress))

        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif name == "step":
        step_size = cfg.get("step_size", 30)
        gamma = cfg.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif name == "none":
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)

    else:
        raise ValueError(f"Unknown scheduler: {name}")
