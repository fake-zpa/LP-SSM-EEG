"""
EEG-Specific Local Modulator — Innovation 1 of LP-SSM-EEG.

Replaces generic label-embedding modulation (original LP-SSM) with
EEG-intrinsic signals derived from frequency, spatial, and event-confidence features.

Three components:
  A. Frequency-band consistency bias: deviation of current block's freq power from running mean
  B. Cross-channel spatial coherence bias: inter-channel correlation deviation
  C. Event-confidence uncertainty bias: entropy of block's local seizure prediction

These are combined into a scalar modulation weight per block.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class FreqBandConsistencyBias(nn.Module):
    """
    Compute deviation of current block's frequency-band power from a running baseline.

    Input:  x [B, T, d_model]  (block representation)
    Also takes raw_x [B, C, T_block] for frequency computation (optional shortcut)

    Output: bias [B, d_bias]
    """

    BANDS = {
        "delta": (0.5, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta":  (13, 30),
    }

    def __init__(self, d_out: int = 16, sfreq: float = 256.0):
        super().__init__()
        self.sfreq = sfreq
        n_bands = len(self.BANDS)
        self.proj = nn.Linear(n_bands, d_out)

    def _band_power(self, x: torch.Tensor, low: float, high: float) -> torch.Tensor:
        """Approximate band power via variance of bandpass-approximated signal."""
        T = x.shape[-1]
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.sfreq).to(x.device)
        X = torch.fft.rfft(x, dim=-1)
        mask = (freqs >= low) & (freqs < high)
        X_band = X * mask.unsqueeze(0).unsqueeze(0)
        x_band = torch.fft.irfft(X_band, n=T, dim=-1)
        return x_band.var(dim=-1).mean(dim=-1)  # [B]

    def forward(self, raw_x: torch.Tensor) -> torch.Tensor:
        """
        raw_x: [B, C, T_block] — raw EEG segment for this block
        Returns: [B, d_out]
        """
        powers = []
        for (low, high) in self.BANDS.values():
            p = self._band_power(raw_x, low, high)  # [B]
            powers.append(p.unsqueeze(-1))
        powers = torch.cat(powers, dim=-1)  # [B, n_bands]
        powers = F.layer_norm(powers, powers.shape[-1:])
        return self.proj(powers)


class CrossChannelCoherenceBias(nn.Module):
    """
    Compute deviation of cross-channel spatial correlation from a reference baseline.

    Input:  raw_x [B, C, T_block]
    Output: bias [B, d_out]
    """

    def __init__(self, in_channels: int = 23, d_out: int = 16):
        super().__init__()
        self.d_out = d_out
        # Learnable reference correlation pattern
        self.ref_corr = nn.Parameter(torch.eye(in_channels))
        self.proj = nn.Linear(in_channels, d_out)

    def forward(self, raw_x: torch.Tensor) -> torch.Tensor:
        """raw_x: [B, C, T_block] → [B, d_out]"""
        B, C, T = raw_x.shape
        # Normalize channels
        x_norm = raw_x - raw_x.mean(-1, keepdim=True)
        x_norm = x_norm / (x_norm.std(-1, keepdim=True) + 1e-6)
        # Correlation matrix: [B, C, C]
        corr = torch.bmm(x_norm, x_norm.transpose(1, 2)) / T
        # Deviation from reference
        ref = self.ref_corr.unsqueeze(0)
        dev = (corr - ref).abs().mean(dim=-1)  # [B, C] — row-wise mean deviation
        return self.proj(dev)


class EventConfidenceBias(nn.Module):
    """
    Compute uncertainty (entropy) of block-level event prediction.
    High entropy → model is uncertain → stronger local signal.

    Input:  h [B, T, d_model] (block hidden state)
    Output: bias [B, d_out]
    """

    def __init__(self, d_model: int, d_out: int = 16, n_classes: int = 2):
        super().__init__()
        self.probe = nn.Sequential(
            nn.Linear(d_model, 32),
            nn.GELU(),
            nn.Linear(32, n_classes),
        )
        self.proj = nn.Linear(1, d_out)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """h: [B, T, d_model] → [B, d_out]"""
        h_pool = h.mean(dim=1)  # [B, d_model]
        logits = self.probe(h_pool)  # [B, n_classes]
        probs = F.softmax(logits, dim=-1)
        entropy = -(probs * (probs + 1e-8).log()).sum(dim=-1, keepdim=True)  # [B, 1]
        return self.proj(entropy)


class EEGLocalModulator(nn.Module):
    """
    EEG-Specific Local Modulator for LP-SSM-EEG.

    Combines three EEG-intrinsic signals into a single scalar modulation weight
    per block, per sample in the batch.

    This replaces label-embedding modulation from original LP-SSM with
    EEG-internal signals that work at inference time without labels.
    """

    def __init__(
        self,
        d_model: int = 128,
        in_channels: int = 23,
        modulator_dim: int = 64,
        sfreq: float = 256.0,
        freqband_consistency: bool = True,
        cross_channel_coherence: bool = True,
        event_confidence: bool = True,
    ):
        super().__init__()
        self.freqband_consistency = freqband_consistency
        self.cross_channel_coherence = cross_channel_coherence
        self.event_confidence = event_confidence

        d_bias = 16
        total_dim = 0

        if freqband_consistency:
            self.freq_bias = FreqBandConsistencyBias(d_out=d_bias, sfreq=sfreq)
            total_dim += d_bias

        if cross_channel_coherence:
            self.spatial_bias = CrossChannelCoherenceBias(in_channels=in_channels, d_out=d_bias)
            total_dim += d_bias

        if event_confidence:
            self.event_bias = EventConfidenceBias(d_model=d_model, d_out=d_bias)
            total_dim += d_bias

        if total_dim == 0:
            total_dim = d_bias

        self._fusion_hidden = nn.Sequential(
            nn.Linear(total_dim, modulator_dim),
            nn.LayerNorm(modulator_dim),
            nn.GELU(),
        )
        self._fusion_out = nn.Linear(modulator_dim, 1)
        # bias=0 → sigmoid(0)=0.5 → weight=1+0.5*0.5=1.25 (near identity at init)
        nn.init.zeros_(self._fusion_out.bias)
        nn.init.xavier_uniform_(self._fusion_out.weight, gain=0.1)

    def forward(
        self,
        h: torch.Tensor,
        raw_x: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        h:     [B, T, d_model] — block hidden state
        raw_x: [B, C, T_block] — raw EEG for this block (required for freq/spatial)

        Returns: modulation_weight [B, 1] in (0, 1)
        """
        parts = []

        if self.freqband_consistency and raw_x is not None:
            parts.append(self.freq_bias(raw_x))

        if self.cross_channel_coherence and raw_x is not None:
            parts.append(self.spatial_bias(raw_x))

        if self.event_confidence:
            parts.append(self.event_bias(h))

        if not parts:
            return torch.ones(h.shape[0], 1, device=h.device)

        combined = torch.cat(parts, dim=-1)
        h_fuse = self._fusion_hidden(combined)
        # Output range [1.0, 1.5]: identity=1.0, stronger=1.5
        return 1.0 + 0.5 * torch.sigmoid(self._fusion_out(h_fuse))
