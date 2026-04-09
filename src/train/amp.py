"""AMP (Automatic Mixed Precision) utilities for 4090 single-card training."""
import torch
from contextlib import contextmanager
from typing import Optional


class AMPContext:
    """Wraps GradScaler and autocast for cleaner training code."""

    def __init__(self, enabled: bool = True, dtype: str = "float16"):
        self.enabled = enabled and torch.cuda.is_available()
        self.dtype = torch.float16 if dtype == "float16" else torch.bfloat16
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.enabled)

    @contextmanager
    def autocast(self):
        with torch.cuda.amp.autocast(enabled=self.enabled, dtype=self.dtype):
            yield

    def scale(self, loss):
        return self.scaler.scale(loss)

    def step(self, optimizer):
        self.scaler.step(optimizer)

    def update(self):
        self.scaler.update()

    def unscale_(self, optimizer):
        if self.enabled:
            self.scaler.unscale_(optimizer)
