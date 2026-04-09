"""
EEG-Specific Local Losses for LP-SSM-EEG.

Structured interface for all losses:
  - Main task loss (cross-entropy with class weighting)
  - Local TF reconstruction loss (per block)
  - Local temporal consistency loss (per block)
  - Event-aware weighting loss (placeholder)
  - Patient-level consistency loss (placeholder)

Total: L = L_main + lambda_local * sum_blocks(L_local^l)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class WeightedCrossEntropyLoss(nn.Module):
    """Cross-entropy with optional class weights for imbalanced datasets."""

    def __init__(self, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, weight=self.class_weights)


class FocalLoss(nn.Module):
    """Focal loss for severe class imbalance (CHB-MIT ~1-2% ictal)."""

    def __init__(self, gamma: float = 2.0, class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor,
                sample_weights: Optional[torch.Tensor] = None) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.class_weights, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1 - pt) ** self.gamma) * ce
        if sample_weights is not None:
            focal = focal * sample_weights.to(focal.device)
        return focal.mean()


class LPSSMLoss(nn.Module):
    """
    Combined LP-SSM-EEG training objective.

    L_total = L_main + lambda_local * (1/N) * sum_i(L_local^i)

    where L_local^i is computed by the LocalDenoisingHead of block i.
    """

    def __init__(
        self,
        n_classes: int = 2,
        class_weights: Optional[torch.Tensor] = None,
        main_loss_type: str = "focal",      # focal | ce
        focal_gamma: float = 2.0,
        local_loss_weight: float = 0.15,
        training_mode: str = "local",       # local | global
    ):
        super().__init__()
        self.local_loss_weight = local_loss_weight
        self.training_mode = training_mode

        if main_loss_type == "focal":
            self.main_loss_fn = FocalLoss(gamma=focal_gamma, class_weights=class_weights)
        else:
            self.main_loss_fn = WeightedCrossEntropyLoss(class_weights=class_weights)

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        local_losses: Optional[List[torch.Tensor]] = None,
        mod_weight: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        logits:       [B, n_classes]
        targets:      [B]
        local_losses: list of per-block scalar losses (from LocalDenoisingHead)
        mod_weight:   [B] optional per-sample weight for main task loss (E06 global+mod)

        Returns dict with: total, main, local
        """
        if mod_weight is not None and hasattr(self.main_loss_fn, 'forward'):
            l_main = self.main_loss_fn(logits, targets, sample_weights=mod_weight)
        else:
            l_main = self.main_loss_fn(logits, targets)

        l_local = torch.tensor(0.0, device=logits.device)
        if self.training_mode == "local" and local_losses:
            valid = [l for l in local_losses if l is not None and torch.isfinite(l)]
            if valid:
                l_local = torch.stack(valid).mean()

        total = l_main + self.local_loss_weight * l_local

        return {
            "loss": total,
            "loss_main": l_main.detach(),
            "loss_local": l_local.detach(),
        }
