"""EEGNet baseline model."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGNet(nn.Module):
    """
    EEGNet: Compact CNN for EEG-based BCIs.
    Input: [B, C, T]  Output: [B, n_classes]
    """

    def __init__(
        self,
        in_channels: int = 23,
        n_classes: int = 2,
        sfreq: float = 256.0,
        window_samples: int = 1024,
        F1: int = 8,
        D: int = 2,
        F2: int = 16,
        dropout: float = 0.5,
        kernel_length: int = 64,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        F2 = F1 * D

        self.block1 = nn.Sequential(
            nn.Conv2d(1, F1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, kernel_size=(in_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(F2, F2, kernel_size=(1, 16), padding=(0, 8), groups=F2, bias=False),
            nn.Conv2d(F2, F2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 8)),
            nn.Dropout(dropout),
        )

        with torch.no_grad():
            dummy = torch.zeros(1, 1, in_channels, window_samples)
            out = self.block1(dummy)
            out = self.block2(out)
            flat_dim = out.view(1, -1).shape[1]

        self.classifier = nn.Linear(flat_dim, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] → unsqueeze to [B, 1, C, T]
        x = x.unsqueeze(1)
        x = self.block1(x)
        x = self.block2(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)
