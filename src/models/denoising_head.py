"""
EEG-Specific Local Denoising Heads — Innovation 2 of LP-SSM-EEG.

Two objectives:
  A. TF Reconstruction Head: reconstruct short-time Fourier spectrum of input block
  B. Temporal Consistency Head: predict that adjacent-window representations should be close
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class TFReconstructionHead(nn.Module):
    """
    Time-Frequency Reconstruction Head.

    Given block hidden state h, predict the STFT magnitude of the corresponding raw EEG.
    L_TF = MSE(pred_TF, target_TF)

    This forces each block to preserve local frequency information.
    """

    def __init__(
        self,
        d_model: int = 128,
        in_channels: int = 23,
        n_fft: int = 64,
        hop_length: int = 16,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        n_freqs = n_fft // 2 + 1

        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, in_channels * n_freqs),
        )
        self.in_channels = in_channels
        self.n_freqs = n_freqs

    def compute_target(self, raw_x: torch.Tensor) -> torch.Tensor:
        """
        raw_x: [B, C, T]
        Returns target TF magnitude: [B, C, n_freqs]
        (averaged over time frames)
        """
        B, C, T = raw_x.shape
        target_list = []
        for c in range(C):
            stft = torch.stft(
                raw_x[:, c, :],
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                return_complex=True,
                window=torch.hann_window(self.n_fft, device=raw_x.device),
            )  # [B, n_freqs, n_frames]
            mag = stft.abs().mean(dim=-1)  # [B, n_freqs]
            target_list.append(mag)
        return torch.stack(target_list, dim=1)  # [B, C, n_freqs]

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """
        h: [B, T, d_model]
        Returns pred_TF: [B, C, n_freqs]
        """
        h_pool = h.mean(dim=1)  # [B, d_model]
        pred = self.head(h_pool)  # [B, C * n_freqs]
        return pred.view(h.shape[0], self.in_channels, self.n_freqs)

    def loss(self, h: torch.Tensor, raw_x: torch.Tensor) -> torch.Tensor:
        """Compute TF reconstruction MSE loss."""
        pred = self.forward(h)
        with torch.no_grad():
            target = self.compute_target(raw_x)
        target = F.layer_norm(target, target.shape[-2:])
        pred = F.layer_norm(pred, pred.shape[-2:])
        return F.mse_loss(pred, target.detach())


class BandSelectiveReconstructionHead(nn.Module):
    """
    EEG-Specific Band-Selective Reconstruction Head (replaces broken TC head).

    For each SSM block, predict per-band EEG power from the block's pooled hidden state.
    Bands are weighted by clinical relevance to seizure detection:
      delta(0.5-4Hz): +2.0  (high-amplitude ictal waves)
      theta(4-8Hz):   +2.0  (ictal onset rhythms)
      alpha(8-13Hz):  +0.5  (suppressed during ictal -> diagnostic)
      beta(13-30Hz):  +2.0  (pre-ictal fast activity)

    This is GENUINELY EEG-specific: it forces each SSM block to maintain
    a representation that tracks the power of clinically relevant EEG bands,
    weighted by their diagnostic value for seizure detection.

    L_band = sum_b w_b * MSE(pred_b(h), bandpower_b(x))
    """

    BANDS = [
        ('delta', 0.5,  4.0,  2.0),
        ('theta', 4.0,  8.0,  2.0),
        ('alpha', 8.0,  13.0, 0.5),
        ('beta',  13.0, 30.0, 2.0),
    ]

    def __init__(self, d_model: int = 128, sfreq: float = 256.0):
        super().__init__()
        self.sfreq = sfreq
        n_bands = len(self.BANDS)
        self.weights = torch.tensor([b[3] for b in self.BANDS])
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, n_bands),
        )

    def _bandpower(self, raw_x: torch.Tensor, low: float, high: float) -> torch.Tensor:
        """raw_x: [B, C, T] -> [B] mean band power across channels"""
        B, C, T = raw_x.shape
        freqs = torch.fft.rfftfreq(T, d=1.0 / self.sfreq, device=raw_x.device)
        X = torch.fft.rfft(raw_x.mean(dim=1), dim=-1)   # [B, T//2+1]
        psd = X.abs().pow(2) / T
        mask = (freqs >= low) & (freqs < high)
        return (psd[:, mask].mean(dim=-1) + 1e-9).log()  # [B]

    def loss(self, h: torch.Tensor, raw_x: torch.Tensor) -> torch.Tensor:
        """
        h:     [B, T, d_model]
        raw_x: [B, C, T]
        Returns: weighted band reconstruction loss (scalar)
        """
        if raw_x is None:
            return torch.tensor(0.0, device=h.device)

        h_pool = h.mean(dim=1)         # [B, d_model]
        pred = self.head(h_pool)        # [B, n_bands]

        targets = []
        for _, lo, hi, _ in self.BANDS:
            targets.append(self._bandpower(raw_x, lo, hi))
        target = torch.stack(targets, dim=-1)  # [B, n_bands]

        target = F.layer_norm(target, target.shape[-1:])
        pred   = F.layer_norm(pred,   pred.shape[-1:])

        band_losses = F.mse_loss(pred, target.detach(), reduction='none').mean(0)  # [n_bands]
        w = self.weights.to(h.device)
        return (w * band_losses).sum() / w.sum()


class LocalDenoisingHead(nn.Module):
    """
    Combined local denoising head per Mamba block.

    L_local = tf_weight * L_TF  +  band_weight * L_band

    L_TF   : TF reconstruction (full STFT spectrum MSE)
    L_band : Band-selective reconstruction (delta/theta/alpha/beta power prediction,
             seizure-weighted) — EEG-specific replacement for the broken TC head
    """

    def __init__(
        self,
        d_model: int = 128,
        in_channels: int = 23,
        n_fft: int = 64,
        hop_length: int = 16,
        tf_weight: float = 1.0,
        consistency_weight: float = 0.5,   # now = band_weight (kept for compat)
        tf_enabled: bool = True,
        consistency_enabled: bool = True,   # now = band_selective_enabled
        sfreq: float = 256.0,
    ):
        super().__init__()
        self.tf_weight = tf_weight
        self.band_weight = consistency_weight
        self.tf_enabled = tf_enabled
        self.band_enabled = consistency_enabled

        if tf_enabled:
            self.tf_head = TFReconstructionHead(d_model, in_channels, n_fft, hop_length)
        if consistency_enabled:
            self.band_head = BandSelectiveReconstructionHead(d_model, sfreq=sfreq)

    def forward(
        self,
        h: torch.Tensor,
        raw_x: Optional[torch.Tensor] = None,
        modulation_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined local denoising loss for this block.

        h:                  [B, T, d_model]
        raw_x:              [B, C, T_block] (needed for TF loss)
        modulation_weight:  [B, 1] scalar weight from local modulator

        Returns: scalar loss
        """
        loss = torch.tensor(0.0, device=h.device)

        if self.tf_enabled and raw_x is not None:
            l_tf = self.tf_head.loss(h, raw_x)
            loss = loss + self.tf_weight * l_tf

        if self.band_enabled and raw_x is not None:
            l_band = self.band_head.loss(h, raw_x)
            loss = loss + self.band_weight * l_band

        if modulation_weight is not None:
            loss = loss * modulation_weight.mean()

        return loss
