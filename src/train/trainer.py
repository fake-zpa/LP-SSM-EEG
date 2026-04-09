"""
Main training loop for LP-SSM-EEG experiments.
Supports: AMP, gradient accumulation, gradient checkpointing, logging, checkpointing.
"""
import json
import time
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.train.amp import AMPContext
from src.train.early_stopping import EarlyStopping
from src.utils.profiler import MemoryProfiler
from src.utils.io import save_json, append_jsonl
from src.utils.reproducibility import generate_run_id, save_run_manifest

logger = logging.getLogger(__name__)


class Trainer:
    """
    Unified trainer for all models (EEGNet, CNN, Transformer, Mamba, LP-SSM-EEG).
    Handles local-loss models (LP-SSM-EEG) and standard models uniformly.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_fn,
        optimizer,
        scheduler=None,
        amp_ctx: Optional[AMPContext] = None,
        max_epochs: int = 100,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
        early_stopping: Optional[EarlyStopping] = None,
        checkpoint_dir: str = "outputs/checkpoints",
        log_dir: str = "logs/train",
        metrics_dir: str = "outputs/metrics",
        run_id: str = None,
        device: str = "cuda",
        log_every_n_steps: int = 10,
        local_loss_warmup_epochs: int = 0,
    ):
        self.model = model
        self.loss_fn = loss_fn
        self.local_loss_warmup_epochs = local_loss_warmup_epochs
        self._base_local_loss_weight = getattr(loss_fn, "local_loss_weight", 0.0)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.amp_ctx = amp_ctx or AMPContext(enabled=False)
        self.max_epochs = max_epochs
        self.grad_accum = gradient_accumulation_steps
        self.max_grad_norm = max_grad_norm
        self.early_stopping = early_stopping
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.log_every_n_steps = log_every_n_steps

        self.run_id = run_id or generate_run_id()
        self.checkpoint_dir = Path(checkpoint_dir) / self.run_id
        self.log_dir = Path(log_dir) / self.run_id
        self.metrics_dir = Path(metrics_dir) / self.run_id

        for d in [self.checkpoint_dir, self.log_dir, self.metrics_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self.profiler = MemoryProfiler()
        self.history: List[Dict] = []
        self.best_val_metric = -float("inf")
        self.global_step = 0

        self.model.to(self.device)

    def _set_local_loss_scale(self, epoch: int):
        """Linearly ramp local loss weight from 0 to base over warmup epochs."""
        if self.local_loss_warmup_epochs > 0 and hasattr(self.loss_fn, "local_loss_weight"):
            scale = min(1.0, epoch / max(1, self.local_loss_warmup_epochs))
            self.loss_fn.local_loss_weight = self._base_local_loss_weight * scale

    def _forward_step(self, batch) -> Dict[str, torch.Tensor]:
        """Forward pass + loss computation. Handles both standard and LP-SSM-EEG models."""
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)

        with self.amp_ctx.autocast():
            out = self.model(x)

            if isinstance(out, dict):
                logits = out["logits"]
                local_losses = out.get("local_losses")
                mod_weight = out.get("mod_weight")
            else:
                logits = out
                local_losses = None
                mod_weight = None

            loss_dict = self.loss_fn(logits, y, local_losses=local_losses, mod_weight=mod_weight)

        return {**loss_dict, "logits": logits, "targets": y}

    def train_epoch(self, loader: DataLoader, epoch: int) -> Dict[str, float]:
        self.model.train()
        self.optimizer.zero_grad()
        self.profiler.reset_peak()

        total_loss = 0.0
        total_main = 0.0
        total_local = 0.0
        n_correct = 0
        n_total = 0
        step_in_epoch = 0

        pbar = tqdm(loader, desc=f"Epoch {epoch}", leave=False)
        for step, batch in enumerate(pbar):
            result = self._forward_step(batch)
            loss = result["loss"] / self.grad_accum

            scaled_loss = self.amp_ctx.scale(loss)
            scaled_loss.backward()

            if (step + 1) % self.grad_accum == 0 or (step + 1) == len(loader):
                self.amp_ctx.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.amp_ctx.step(self.optimizer)
                self.amp_ctx.update()
                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1

            total_loss += result["loss"].item()
            total_main += result["loss_main"].item()
            total_local += result["loss_local"].item()
            preds = result["logits"].argmax(dim=-1)
            n_correct += (preds == result["targets"]).sum().item()
            n_total += len(result["targets"])
            step_in_epoch += 1

            if step % self.log_every_n_steps == 0:
                pbar.set_postfix({
                    "loss": f"{result['loss'].item():.4f}",
                    "acc": f"{n_correct/max(n_total,1):.3f}",
                })

        n = max(step_in_epoch, 1)
        return {
            "train_loss": total_loss / n,
            "train_loss_main": total_main / n,
            "train_loss_local": total_local / n,
            "train_acc": n_correct / max(n_total, 1),
            "peak_vram_gb": self.profiler.peak_vram_gb() or 0.0,
        }

    @torch.no_grad()
    def eval_epoch(self, loader: DataLoader) -> Dict[str, float]:
        self.model.eval()
        all_logits = []
        all_targets = []
        total_loss = 0.0
        n_steps = 0

        for batch in tqdm(loader, desc="Eval", leave=False):
            x, y = batch
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            with self.amp_ctx.autocast():
                out = self.model(x)
                logits = out["logits"] if isinstance(out, dict) else out
                ld = self.loss_fn(logits, y, local_losses=None)
                total_loss += ld["loss"].item()
                n_steps += 1

            all_logits.append(logits.float().cpu())
            all_targets.append(y.cpu())

        from src.eval.metrics_classification import compute_metrics
        logits_cat = torch.cat(all_logits, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)
        metrics = compute_metrics(logits_cat, targets_cat)
        metrics["val_loss"] = total_loss / max(n_steps, 1)
        return metrics

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        monitor: str = "val_auroc",
    ) -> Dict:
        logger.info(f"Training run_id={self.run_id} for {self.max_epochs} epochs on {self.device}")
        t0 = time.time()

        for epoch in range(1, self.max_epochs + 1):
            self._set_local_loss_scale(epoch)
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.eval_epoch(val_loader)

            epoch_metrics = {
                "epoch": epoch,
                "lr": self.optimizer.param_groups[0]["lr"],
                **train_metrics,
                **val_metrics,
            }
            self.history.append(epoch_metrics)
            append_jsonl(epoch_metrics, self.metrics_dir / "epoch_metrics.jsonl")

            val_score = val_metrics.get(monitor, val_metrics.get("val_auroc", 0.0))
            logger.info(
                f"Epoch {epoch:3d} | "
                f"loss={train_metrics['train_loss']:.4f} | "
                f"val_auroc={val_metrics.get('val_auroc', 0):.4f} | "
                f"val_auprc={val_metrics.get('val_auprc', 0):.4f} | "
                f"VRAM={train_metrics.get('peak_vram_gb', 0):.1f}GB"
            )

            if val_score > self.best_val_metric:
                self.best_val_metric = val_score
                self._save_checkpoint("best.pt", epoch_metrics)

            self._save_checkpoint("last.pt", epoch_metrics)

            if self.early_stopping and self.early_stopping.step(val_score):
                logger.info(f"Early stopping at epoch {epoch}")
                break

        elapsed = time.time() - t0
        summary = {
            "run_id": self.run_id,
            "total_epochs": epoch,
            "best_val_metric": self.best_val_metric,
            "training_time_min": round(elapsed / 60, 2),
            "best_metrics": self.history[
                max(0, next((i for i, h in enumerate(self.history) if h.get(monitor, 0) == self.best_val_metric), -1))
            ],
        }
        save_json(summary, self.metrics_dir / "training_summary.json")
        logger.info(f"Training complete. Best {monitor}={self.best_val_metric:.4f}, time={elapsed/60:.1f}min")
        return summary

    def _save_checkpoint(self, name: str, metrics: dict):
        # Unwrap torch.compile wrapper (_orig_mod) before saving
        model_to_save = getattr(self.model, "_orig_mod", self.model)
        ckpt = {
            "model_state": model_to_save.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "metrics": metrics,
            "run_id": self.run_id,
        }
        torch.save(ckpt, self.checkpoint_dir / name)
