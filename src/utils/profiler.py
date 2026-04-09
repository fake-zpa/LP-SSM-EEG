"""Memory and timing profiling utilities for 4090 single-card training."""
import time
from contextlib import contextmanager
from typing import Dict, Optional


class MemoryProfiler:
    """Track GPU/CPU memory usage during training."""

    def __init__(self):
        self.records: Dict[str, float] = {}

    def snapshot(self, tag: str) -> Dict[str, float]:
        result = {"tag": tag}
        try:
            import torch
            if torch.cuda.is_available():
                result["gpu_allocated_gb"] = round(
                    torch.cuda.memory_allocated() / 1e9, 3
                )
                result["gpu_reserved_gb"] = round(
                    torch.cuda.memory_reserved() / 1e9, 3
                )
                result["gpu_peak_allocated_gb"] = round(
                    torch.cuda.max_memory_allocated() / 1e9, 3
                )
        except ImportError:
            pass
        try:
            import psutil, os
            proc = psutil.Process(os.getpid())
            result["cpu_rss_gb"] = round(proc.memory_info().rss / 1e9, 3)
        except ImportError:
            pass
        self.records[tag] = result
        return result

    def reset_peak(self):
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()
        except ImportError:
            pass

    def peak_vram_gb(self) -> Optional[float]:
        try:
            import torch
            if torch.cuda.is_available():
                return round(torch.cuda.max_memory_allocated() / 1e9, 3)
        except ImportError:
            pass
        return None


@contextmanager
def timer(name: str = "", logger=None):
    """Context manager for timing a code block."""
    t0 = time.perf_counter()
    yield
    elapsed = time.perf_counter() - t0
    msg = f"[TIMER] {name}: {elapsed:.3f}s"
    if logger is not None:
        logger.info(msg)
    else:
        print(msg)
