"""1D CNN Baseline for EEG classification."""
import torch
import torch.nn as nn


class ResBlock1D(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 7, dropout: float = 0.2):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(channels, channels, kernel_size, padding=pad, bias=False),
            nn.BatchNorm1d(channels),
        )
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv(x))


class CNNBaseline(nn.Module):
    """
    1D CNN with residual blocks over temporal dimension.
    Input: [B, C, T]  Output: [B, n_classes]
    """

    def __init__(
        self,
        in_channels: int = 23,
        n_classes: int = 2,
        base_filters: int = 64,
        n_blocks: int = 4,
        kernel_size: int = 7,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, base_filters, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(base_filters),
            nn.GELU(),
        )

        self.blocks = nn.Sequential(
            *[ResBlock1D(base_filters, kernel_size, dropout) for _ in range(n_blocks)]
        )

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(base_filters, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.pool(x).squeeze(-1)
        return self.classifier(x)
