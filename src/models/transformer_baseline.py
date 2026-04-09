"""Transformer Baseline for EEG classification."""
import math
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class TransformerBaseline(nn.Module):
    """
    Standard Transformer over EEG time steps.
    Input: [B, C, T]  Output: [B, n_classes]
    Treats each time step as a token (after channel projection).
    """

    def __init__(
        self,
        in_channels: int = 23,
        n_classes: int = 2,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 256,
        dropout: float = 0.1,
        max_seq_len: int = 1024,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len + 1, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] → transpose → [B, T, C]
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)  # [B, T, d_model]

        cls = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, T+1, d_model]
        x = self.pos_enc(x)
        x = self.encoder(x)
        x = self.norm(x[:, 0])  # CLS token
        return self.classifier(x)
