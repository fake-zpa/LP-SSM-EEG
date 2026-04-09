"""Optimizer factory."""
import torch
from typing import Any, Dict


def build_optimizer(model: torch.nn.Module, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    name = cfg.get("name", "adamw").lower()
    lr = cfg.get("lr", 1e-3)
    wd = cfg.get("weight_decay", 1e-4)
    betas = tuple(cfg.get("betas", [0.9, 0.999]))
    eps = cfg.get("eps", 1e-8)

    params = [p for p in model.parameters() if p.requires_grad]

    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)
    elif name == "adam":
        return torch.optim.Adam(params, lr=lr, weight_decay=wd, betas=betas, eps=eps)
    elif name == "sgd":
        momentum = cfg.get("momentum", 0.9)
        return torch.optim.SGD(params, lr=lr, weight_decay=wd, momentum=momentum)
    else:
        raise ValueError(f"Unknown optimizer: {name}")
