"""
LP-SSM-EEG: Local-Plasticity Selective State Space Learning for Long-Context EEG.

This is the core method of the paper. It extends MambaBaseline with:
  1. EEG-specific Local Modulator (freq-band, cross-channel, event-confidence)
  2. EEG-specific Local Denoising Heads (TF reconstruction + temporal consistency)
  3. Configurable training mode: local | global (for clean mechanism ablation)

Architecture:
    Input EEG [B, C, T]
      ↓ Input Projection
      ↓ × N LP-SSM Blocks:
         ├── Selective SSM Core (MambaBlock)
         ├── EEG Local Modulator  ← Innovation 1
         └── Local Denoising Head ← Innovation 2
      ↓ Main Task Classifier

Training:
    L_total = L_main + λ × (1/N) × Σ_l L_local^l
    
    mode=global: L_local not computed; gradients flow globally (= Mamba baseline training)
    mode=local:  L_local computed per block; optional detach between blocks
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple

from src.models.mamba_baseline import MambaBlock
from src.models.local_modulator import EEGLocalModulator
from src.models.local_modulator_v2 import EEGLocalModulatorV2
from src.models.denoising_head import LocalDenoisingHead
from src.models.losses import LPSSMLoss


class LPSSMBlock(nn.Module):
    """
    One LP-SSM-EEG block: MambaBlock + EEG Local Modulator + Local Denoising Head.
    """

    def __init__(
        self,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
        in_channels: int = 22,
        sfreq: float = 256.0,
        n_fft: int = 64,
        hop_length: int = 16,
        tf_weight: float = 1.0,
        consistency_weight: float = 0.5,
        tf_enabled: bool = True,
        consistency_enabled: bool = True,
        mod_use_band_powers: bool = True,
        mod_use_ictal_ratio: bool = True,
        mod_use_temporal_variance: bool = True,
        mod_use_event_uncertainty: bool = True,
    ):
        super().__init__()
        self.ssm = MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)

        self.modulator = EEGLocalModulatorV2(
            d_model=d_model,
            in_channels=in_channels,
            sfreq=sfreq,
            use_band_powers=mod_use_band_powers,
            use_ictal_ratio=mod_use_ictal_ratio,
            use_temporal_variance=mod_use_temporal_variance,
            use_event_uncertainty=mod_use_event_uncertainty,
        )

        self.local_head = LocalDenoisingHead(
            d_model=d_model,
            in_channels=in_channels,
            n_fft=n_fft,
            hop_length=hop_length,
            tf_weight=tf_weight,
            consistency_weight=consistency_weight,
            tf_enabled=tf_enabled,
            consistency_enabled=consistency_enabled,
            sfreq=sfreq,
        )

    def forward(
        self,
        x: torch.Tensor,
        raw_x: Optional[torch.Tensor] = None,
        compute_local_loss: bool = True,
        detach_input: bool = False,
        use_modulator: bool = True,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        x:                  [B, T, d_model]
        raw_x:              [B, C, T_block] or None
        compute_local_loss: if True, compute local denoising loss
        detach_input:       if True, detach input from prev block (fully local learning)

        Returns:
            h:           [B, T, d_model] (block output)
            local_loss:  scalar or None
        """
        if detach_input:
            x = x.detach()

        h = self.ssm(x)

        local_loss = None
        if compute_local_loss:
            mod_weight = self.modulator(h, raw_x=raw_x) if use_modulator else None
            local_loss = self.local_head(h, raw_x=raw_x, modulation_weight=mod_weight)

        return h, local_loss


class LPSSMEEG(nn.Module):
    """
    LP-SSM-EEG: Main method model.

    Input:  [B, C, T]
    Output: [B, n_classes]

    Supports two training modes (via training_mode config):
      'global': no local losses computed → equivalent to MambaBaseline training
      'local' : local losses computed + optionally detach between blocks
    """

    def __init__(
        self,
        in_channels: int = 22,
        n_classes: int = 2,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 4,
        dropout: float = 0.1,
        sfreq: float = 256.0,
        n_fft: int = 64,
        hop_length: int = 16,
        tf_reconstruction_enabled: bool = True,
        tf_weight: float = 1.0,
        temporal_consistency_enabled: bool = True,
        consistency_weight: float = 0.5,
        training_mode: str = "local",
        detach_between_blocks: bool = False,
        use_modulator: bool = True,
        use_main_loss_mod: bool = False,
        mod_use_band_powers: bool = True,
        mod_use_ictal_ratio: bool = True,
        mod_use_temporal_variance: bool = True,
        mod_use_event_uncertainty: bool = True,
    ):
        super().__init__()
        self.training_mode = training_mode
        self.detach_between_blocks = detach_between_blocks
        self.use_modulator = use_modulator
        self.use_main_loss_mod = use_main_loss_mod
        self.n_layers = n_layers

        self.input_proj = nn.Linear(in_channels, d_model)

        self.blocks = nn.ModuleList([
            LPSSMBlock(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                dropout=dropout,
                in_channels=in_channels,
                sfreq=sfreq,
                n_fft=n_fft,
                hop_length=hop_length,
                tf_weight=tf_weight,
                consistency_weight=consistency_weight,
                tf_enabled=tf_reconstruction_enabled,
                consistency_enabled=temporal_consistency_enabled,
                mod_use_band_powers=mod_use_band_powers,
                mod_use_ictal_ratio=mod_use_ictal_ratio,
                mod_use_temporal_variance=mod_use_temporal_variance,
                mod_use_event_uncertainty=mod_use_event_uncertainty,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(
        self,
        x: torch.Tensor,
        return_local_losses: bool = None,
    ) -> Dict[str, torch.Tensor]:
        """
        x: [B, C, T]

        Returns dict:
            logits:       [B, n_classes]
            local_losses: list of per-block scalar losses (if local mode)
        """
        if return_local_losses is None:
            return_local_losses = (self.training_mode == "local") and self.training

        raw_x = x  # save for local objectives
        h = x.permute(0, 2, 1)   # [B, T, C]
        h = self.input_proj(h)    # [B, T, d_model]

        local_losses = []
        for block in self.blocks:
            h, local_loss = block(
                h,
                raw_x=raw_x,
                compute_local_loss=return_local_losses,
                detach_input=self.detach_between_blocks and return_local_losses,
                use_modulator=self.use_modulator,
            )
            if local_loss is not None:
                local_losses.append(local_loss)

        h = self.norm(h)
        pooled = h.mean(dim=1)
        logits = self.classifier(pooled)

        mod_weight = None
        if self.use_main_loss_mod and self.training_mode == "global" and self.training:
            mod_weight = self._ictal_ratio_weight(raw_x)

        return {
            "logits": logits,
            "local_losses": local_losses if local_losses else None,
            "mod_weight": mod_weight,
        }

    @staticmethod
    def _ictal_ratio_weight(raw_x: torch.Tensor, sfreq: float = 256.0) -> torch.Tensor:
        """Compute per-sample ictal ratio weight in [0.5, 2.0] without learned params.

        ictal_ratio = log((P_beta + P_theta) / (P_alpha + P_delta))
        Maps to [0.5, 2.0] via batch-relative sigmoid normalization.
        """
        B, C, T = raw_x.shape
        freqs = torch.fft.rfftfreq(T, d=1.0 / sfreq, device=raw_x.device)
        X = torch.fft.rfft(raw_x.float().mean(dim=1), dim=-1)
        psd = X.abs().pow(2) / T

        def _band(lo, hi):
            mask = (freqs >= lo) & (freqs < hi)
            return psd[:, mask].mean(dim=-1) + 1e-10

        delta = _band(0.5, 4)
        theta = _band(4, 8)
        alpha = _band(8, 13)
        beta  = _band(13, 30)
        ratio = torch.log((beta + theta) / (alpha + delta))  # [B]
        z = (ratio - ratio.mean()) / (ratio.std() + 1e-6)    # batch-normalize
        return (0.5 + 1.5 * torch.sigmoid(z)).detach()       # [B], no grad through weight
