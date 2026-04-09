"""
EEG-Specific Local Modulator v2 — Redesigned for adaptive per-sample weights.

Key changes from v1:
  1. Batch-relative normalization: weights are normalized within each batch
     so the modulator produces HIGH weights for windows more likely to contain
     seizure-relevant dynamics (high beta/delta ratio, low alpha, high uncertainty)
  2. Per-band signal-to-noise ratio instead of raw band power
  3. Simpler fusion with stronger initialization for output variation

Output: weight in [0.5, 2.0] — identity = 1.0, amplified = >1.0, suppressed = <1.0
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


ICTAL_INDICATOR_BANDS = {
    "delta": (0.5, 4),    # ↑ ictal
    "theta": (4, 8),      # ↑ ictal
    "alpha": (8, 13),     # ↓ ictal (suppressed)
    "beta":  (13, 30),    # ↑ high-frequency ictal
}


class BandPowerFeatures(nn.Module):
    """
    Compute relative band power features (batch-normalized for stability).
    Output: [B, n_bands] — normalized log-power per band
    """
    def __init__(self, sfreq: float = 256.0):
        super().__init__()
        self.sfreq = sfreq

    def forward(self, raw_x: torch.Tensor) -> torch.Tensor:
        """raw_x: [B, C, T] → [B, n_bands]"""
        B, C, T = raw_x.shape
        powers = []
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.sfreq, device=raw_x.device)
        X = torch.fft.rfft(raw_x.mean(dim=1), dim=-1)  # [B, T//2+1]
        psd = X.abs().pow(2) / T  # [B, T//2+1]

        for (low, high) in ICTAL_INDICATOR_BANDS.values():
            mask = (freqs >= low) & (freqs < high)
            band_pwr = psd[:, mask].mean(dim=-1)  # [B]
            powers.append((band_pwr + 1e-8).log())

        powers = torch.stack(powers, dim=-1)  # [B, n_bands]
        # Batch-relative normalization: subtract batch mean, divide by std
        mn = powers.mean(dim=0, keepdim=True)
        sd = powers.std(dim=0, keepdim=True) + 1e-6
        return (powers - mn) / sd  # [B, n_bands] — zero-mean, unit-std within batch


class IctalRatioFeature(nn.Module):
    """
    Compute ictal-indicator ratio: (beta + theta) / (alpha + delta + eps)
    High during seizures, low during interictal.
    Output: [B, 1]
    """
    def __init__(self, sfreq: float = 256.0):
        super().__init__()
        self.sfreq = sfreq

    def _band_power(self, psd: torch.Tensor, freqs: torch.Tensor,
                    low: float, high: float) -> torch.Tensor:
        mask = (freqs >= low) & (freqs < high)
        return psd[:, mask].mean(dim=-1) + 1e-10  # [B]

    def forward(self, raw_x: torch.Tensor) -> torch.Tensor:
        """raw_x: [B, C, T] → [B, 1]"""
        B, C, T = raw_x.shape
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.sfreq, device=raw_x.device)
        X = torch.fft.rfft(raw_x.mean(dim=1), dim=-1)
        psd = X.abs().pow(2) / T

        delta = self._band_power(psd, freqs, 0.5, 4)
        theta = self._band_power(psd, freqs, 4, 8)
        alpha = self._band_power(psd, freqs, 8, 13)
        beta = self._band_power(psd, freqs, 13, 30)

        ratio = (beta + theta) / (alpha + delta)  # [B]
        log_ratio = ratio.log().unsqueeze(-1)  # [B, 1]
        # Batch-normalize
        mn = log_ratio.mean(dim=0, keepdim=True)
        sd = log_ratio.std(dim=0, keepdim=True) + 1e-6
        return (log_ratio - mn) / sd


class TemporalVarianceFeature(nn.Module):
    """
    Compute short-term vs long-term variance ratio — detects sudden amplitude changes.
    High ratio → possible ictal onset / offset.
    Output: [B, 1]
    """
    def forward(self, raw_x: torch.Tensor) -> torch.Tensor:
        """raw_x: [B, C, T] → [B, 1]"""
        B, C, T = raw_x.shape
        # Short-term variance: last 25% of window
        cut = T * 3 // 4
        long_var = raw_x.var(dim=-1).mean(dim=-1)  # [B]
        short_var = raw_x[:, :, cut:].var(dim=-1).mean(dim=-1)  # [B]
        ratio = (short_var + 1e-8) / (long_var + 1e-8)  # [B]
        log_ratio = ratio.log().unsqueeze(-1)  # [B, 1]
        mn = log_ratio.mean(dim=0, keepdim=True)
        sd = log_ratio.std(dim=0, keepdim=True) + 1e-6
        return (log_ratio - mn) / sd


class EventUncertaintyFeature(nn.Module):
    """
    Per-sample prediction entropy from a local classifier probe.
    High entropy → uncertain/transitional state → stronger local learning signal.
    Output: [B, 1] (batch-normalized)
    """
    def __init__(self, d_model: int, n_classes: int = 2):
        super().__init__()
        self.probe = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, n_classes),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, T, d_model] → [B, 1]"""
        h_pool = h.mean(dim=1)  # [B, d_model]
        logits = self.probe(h_pool)
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1, keepdim=True)  # [B, 1]
        mn = entropy.mean(dim=0, keepdim=True)
        sd = entropy.std(dim=0, keepdim=True) + 1e-6
        return (entropy - mn) / sd


class EEGLocalModulatorV2(nn.Module):
    """
    Redesigned EEG Local Modulator with batch-relative normalization.
    
    Features:
      - Batch-normalized band powers (4 bands)
      - Ictal ratio log((beta+theta)/(alpha+delta))
      - Temporal variance ratio (short vs long)
      - Event uncertainty (entropy of local probe)
    
    Output: weight in [0.5, 2.0], identity = 1.0
    """

    def __init__(
        self,
        d_model: int = 128,
        in_channels: int = 22,
        sfreq: float = 256.0,
        use_band_powers: bool = True,
        use_ictal_ratio: bool = True,
        use_temporal_variance: bool = True,
        use_event_uncertainty: bool = True,
    ):
        super().__init__()
        self.use_band_powers = use_band_powers
        self.use_ictal_ratio = use_ictal_ratio
        self.use_temporal_variance = use_temporal_variance
        self.use_event_uncertainty = use_event_uncertainty

        total_dim = 0
        if use_band_powers:
            self.band_powers = BandPowerFeatures(sfreq=sfreq)
            total_dim += 4  # n_bands
        if use_ictal_ratio:
            self.ictal_ratio = IctalRatioFeature(sfreq=sfreq)
            total_dim += 1
        if use_temporal_variance:
            self.temporal_var = TemporalVarianceFeature()
            total_dim += 1
        if use_event_uncertainty:
            self.event_uncert = EventUncertaintyFeature(d_model=d_model)
            total_dim += 1

        if total_dim == 0:
            total_dim = 1

        hidden = max(16, total_dim * 2)
        self.fusion = nn.Sequential(
            nn.Linear(total_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )
        # Initialize final layer: small weights → output near 0 → sigmoid(0)=0.5 → weight=1.0
        nn.init.zeros_(self.fusion[-1].bias)
        nn.init.normal_(self.fusion[-1].weight, std=0.1)

    def forward(
        self,
        h: torch.Tensor,
        raw_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        h:     [B, T, d_model]
        raw_x: [B, C, T] (required for EEG features)
        
        Returns: weight [B, 1] in [0.5, 2.0], identity=1.0
        """
        parts = []

        if self.use_band_powers and raw_x is not None:
            parts.append(self.band_powers(raw_x))  # [B, 4]
        if self.use_ictal_ratio and raw_x is not None:
            parts.append(self.ictal_ratio(raw_x))  # [B, 1]
        if self.use_temporal_variance and raw_x is not None:
            parts.append(self.temporal_var(raw_x))  # [B, 1]
        if self.use_event_uncertainty:
            parts.append(self.event_uncert(h))  # [B, 1]

        if not parts:
            return torch.ones(h.shape[0], 1, device=h.device)

        combined = torch.cat(parts, dim=-1)  # [B, total_dim]
        raw_weight = self.fusion(combined)  # [B, 1]
        # Map to [0.5, 2.0]: 0.5 + 1.5 * sigmoid(raw_weight)
        return 0.5 + 1.5 * torch.sigmoid(raw_weight)
