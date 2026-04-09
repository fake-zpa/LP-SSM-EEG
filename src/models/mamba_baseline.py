"""
Mamba/Selective SSM Baseline — pure-PyTorch implementation.

Implements selective state space recurrence without CUDA kernel dependency.
This is the direct ablation baseline for LP-SSM-EEG:
  same backbone, standard full backpropagation, no local objectives.

Reference: Gu & Dao (2023), "Mamba: Linear-Time Sequence Modeling with Selective State Spaces"
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


def _parallel_scan(a: torch.Tensor, b: torch.Tensor):
    """
    Hillis-Steele inclusive parallel prefix scan for the linear recurrence:
        h_t = a_t * h_{t-1} + b_t   (h_0 = 0)

    Associative operator: (a2, b2) ∘ (a1, b1) = (a2*a1, a2*b1 + b2)

    a, b: [B, L, d_model, d_state]
    Returns: (a_prefix, h_prefix) where h_prefix[:, t] == h_t.
    Complexity: O(log L) sequential GPU kernel launches.
    """
    B, L, d, s = a.shape
    L_pad = 1 << (L - 1).bit_length() if L > 1 else 1

    if L_pad > L:
        pad = L_pad - L
        a = torch.cat([a, torch.ones(B, pad, d, s, device=a.device, dtype=a.dtype)], dim=1)
        b = torch.cat([b, torch.zeros(B, pad, d, s, device=b.device, dtype=b.dtype)], dim=1)

    a_work = a.clone()
    b_work = b.clone()
    stride = 1
    while stride < L_pad:
        # Shift right by stride (positions [stride:] receive influence from [:-stride])
        a_shifted = torch.zeros_like(a_work)
        b_shifted = torch.zeros_like(b_work)
        a_shifted[:, stride:] = a_work[:, :-stride]
        b_shifted[:, stride:] = b_work[:, :-stride]
        # Identity for positions < stride: a_shifted=1, b_shifted=0 (already zero-initialized)
        a_shifted[:, :stride] = 1.0

        # Merge current ∘ shifted: new_a = a*a_s, new_b = a*b_s + b
        a_orig = a_work
        a_work = a_orig * a_shifted
        b_work = a_orig * b_shifted + b_work
        stride *= 2

    return a_work[:, :L], b_work[:, :L]


class SelectiveSSMCore(nn.Module):
    """
    Pure-PyTorch Selective SSM (Mamba-style).

    State update:
        h_t = A_bar_t * h_{t-1} + B_bar_t * x_t
        y_t = C_t * h_t

    where A_bar, B_bar are discretized via ZOH:
        A_bar = exp(Δ * A)
        B_bar = (A_bar - I) * inv(A) * B  [simplified: Δ * B for small Δ]

    A_t, B_t, C_t, Δ_t are all input-dependent (selective).
    """

    def __init__(self, d_model: int, d_state: int = 16, dt_rank: str = "auto"):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank

        # Input-dependent projections (selective mechanism)
        self.x_proj = nn.Linear(d_model, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, d_model, bias=True)

        # Initialize dt_proj bias (controls memory decay timescale)
        # We want softplus(bias) ≈ dt_target, so bias = softplus_inv(dt_target)
        # softplus_inv(y) = log(exp(y) - 1) ≈ log(y) for small y (y << 1)
        dt_init_std = self.dt_rank ** -0.5
        nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        dt_log_min = math.log(0.001)
        dt_log_max = math.log(0.1)
        dt_target = torch.exp(torch.rand(d_model) * (dt_log_max - dt_log_min) + dt_log_min)
        with torch.no_grad():
            # softplus_inv(y) = log(expm1(y)) = log(exp(y)-1); for small y ≈ log(y)
            self.dt_proj.bias.copy_(torch.log(torch.expm1(dt_target).clamp(min=1e-8)))

        # State matrix A (log-parameterized for stability)
        A = torch.arange(1, d_state + 1).float().unsqueeze(0).repeat(d_model, 1)
        self.A_log = nn.Parameter(torch.log(A))

        # Output projection D (skip connection)
        self.D = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, d_model]
        returns: [B, L, d_model]
        """
        B, L, d = x.shape

        # Compute selective parameters from input
        xz = self.x_proj(x)  # [B, L, dt_rank + 2*d_state]
        dt_raw, B_mat, C_mat = torch.split(
            xz, [self.dt_rank, self.d_state, self.d_state], dim=-1
        )

        dt = F.softplus(self.dt_proj(dt_raw)).clamp(min=1e-4, max=10.0)  # [B, L, d_model]
        A = -torch.exp(self.A_log.float()).clamp(max=-1e-4)             # [d_model, d_state] (negative)

        # Discretize: simplified ZOH (Δ * A, Δ * B)
        # A_bar: [B, L, d_model, d_state]
        dt_unsq = dt.unsqueeze(-1)
        A_bar = torch.exp(dt_unsq * A.unsqueeze(0).unsqueeze(0)).clamp(max=1.0)
        B_bar = dt_unsq * B_mat.unsqueeze(2)  # [B, L, d_model, d_state]

        # Parallel associative scan: h_t = a_t * h_{t-1} + b_t
        # a_t = A_bar[:,t], b_t = B_bar[:,t] * x[:,t].unsqueeze(-1)
        # Merge operator: (a2, b2) ∘ (a1, b1) = (a2*a1, a2*b1 + b2)
        # [B, L, d_model, d_state]
        a = A_bar  # [B, L, d, s]
        b = B_bar * x.unsqueeze(-1)  # [B, L, d, s]

        a, b = _parallel_scan(a, b)  # returns cumulative state at each step

        # y_t = C_t * h_t  (h_t stored in b after scan)
        y = (b * C_mat.unsqueeze(2)).sum(-1)   # [B, L, d_model]
        y = y + x * self.D.unsqueeze(0).unsqueeze(0)

        return y


try:
    from mamba_ssm import Mamba as _MambaSSM
    MAMBA_SSM_AVAILABLE = True
except ImportError:
    MAMBA_SSM_AVAILABLE = False


class MambaBlock(nn.Module):
    """
    One Mamba block with automatic backend selection:
      - mamba_ssm CUDA kernel  (fast, memory-efficient) when mamba-ssm is installed
      - Pure-PyTorch fallback  (portable, higher VRAM) otherwise

    For paper: uses CUDA kernel backend for efficiency experiments (Table 4).
    Pure-PyTorch fallback used for portability verification only.
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        if MAMBA_SSM_AVAILABLE:
            # Official Mamba CUDA kernel — handles in_proj, conv, ssm, out_proj internally
            self._mamba = _MambaSSM(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
            )
            self._use_cuda = True
        else:
            # Pure-PyTorch fallback
            d_inner = d_model * expand
            self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)
            self.conv1d = nn.Conv1d(
                d_inner, d_inner, kernel_size=d_conv,
                padding=d_conv - 1, groups=d_inner, bias=True,
            )
            self.ssm = SelectiveSSMCore(d_inner, d_state=d_state)
            self.out_proj = nn.Linear(d_inner, d_model, bias=False)
            self._use_cuda = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm(x)

        if self._use_cuda:
            # mamba_ssm.Mamba handles residual internally — we subtract it back
            x = self._mamba(x)
        else:
            xz = self.in_proj(x)
            x_inner, z = xz.chunk(2, dim=-1)
            x_inner = rearrange(x_inner, "b l d -> b d l")
            x_inner = self.conv1d(x_inner)[..., : x.shape[1]]
            x_inner = rearrange(x_inner, "b d l -> b l d")
            x_inner = F.silu(x_inner)
            x_inner = self.ssm(x_inner)
            x_inner = x_inner * F.silu(z)
            x = self.out_proj(x_inner)

        x = self.drop(x)
        return x + residual


class MambaBaseline(nn.Module):
    """
    Standard Mamba/Selective SSM baseline for EEG classification.
    Full global backpropagation — no local objectives.
    This is the direct comparison baseline for LP-SSM-EEG.

    Input: [B, C, T]  Output: [B, n_classes]
    """

    def __init__(
        self,
        in_channels: int = 23,
        n_classes: int = 2,
        d_model: int = 128,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        n_layers: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(in_channels, d_model)
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state=d_state, d_conv=d_conv, expand=expand, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.pool = "mean"
        self.classifier = nn.Linear(d_model, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T] → [B, T, d_model]
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        x = x.mean(dim=1)  # mean pooling
        return self.classifier(x)

    def get_block_representations(self, x: torch.Tensor):
        """Return intermediate representations after each block (for local objectives)."""
        x = x.permute(0, 2, 1)
        x = self.input_proj(x)
        reps = []
        for block in self.blocks:
            x = block(x)
            reps.append(x)
        return reps, self.norm(x)
