"""
Measure GPU inference time and parameter count for all models.
Outputs: ms per window (single sample) and ms per batch (256 windows).
"""
import sys, time, json
sys.path.insert(0, ".")
import torch
import numpy as np
from pathlib import Path

DEVICE = "cuda"
BATCH_SIZES = [1, 256]
N_WARMUP   = 50
N_TRIALS   = 200
SEQ_LEN    = 1024   # 4s @ 256Hz
N_CHANNELS = 23     # CHB-MIT channels used

results = {}

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure(model, batch_size, n_warmup=N_WARMUP, n_trials=N_TRIALS):
    model.eval()
    x = torch.randn(batch_size, N_CHANNELS, SEQ_LEN, device=DEVICE, dtype=torch.float32)
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = model(x)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_trials):
            _ = model(x)
    torch.cuda.synchronize()
    elapsed = (time.perf_counter() - t0) / n_trials * 1000  # ms per call
    return elapsed

# ── Load configs ────────────────────────────────────────────────────────────
import yaml
from src.models.lp_ssm_eeg  import LPSSMEEG
from src.models.mamba_baseline import MambaBaseline

MAMBA_BASE_KW = dict(in_channels=23, n_classes=2, d_model=256, d_state=16,
                     d_conv=4, expand=2, n_layers=6, dropout=0.1)

LPSM_KW = dict(**MAMBA_BASE_KW,
               sfreq=256, n_fft=256, hop_length=64,
               tf_reconstruction_enabled=True, tf_weight=1.0,
               temporal_consistency_enabled=False, consistency_weight=0.0,
               training_mode=False, detach_between_blocks=False,
               use_modulator=True, use_main_loss_mod=False,
               mod_use_band_powers=False, mod_use_ictal_ratio=True,
               mod_use_temporal_variance=False, mod_use_event_uncertainty=False)

model_cfgs = {
    "Mamba (2.64M)":       (MambaBaseline, MAMBA_BASE_KW),
    "LP-SSM-EEG (4.40M)":  (LPSSMEEG,     LPSM_KW),
}

print(f"{'Model':30s} | {'Params':>10s} | {'Latency (bs=1)':>15s} | {'Throughput (bs=256)':>20s}")
print("-" * 85)

for name, (ModelCls, kw) in model_cfgs.items():
    # Build model
    try:
        model = ModelCls(**kw).to(DEVICE).eval()
    except Exception as e:
        print(f"  {name}: build error: {e}")
        continue

    n_params = count_params(model)
    lat_bs1  = measure(model, 1)
    lat_bs256 = measure(model, 256)
    throughput = 256 / (lat_bs256 / 1000)   # windows / sec

    print(f"{name:30s} | {n_params/1e6:>8.2f}M | {lat_bs1:>12.2f} ms | "
          f"{lat_bs256:>10.2f} ms/batch  ({throughput:,.0f} win/s)")

    results[name] = {
        "params_M": round(n_params / 1e6, 3),
        "latency_bs1_ms": round(lat_bs1, 3),
        "latency_bs256_ms": round(lat_bs256, 3),
        "throughput_wps": round(throughput, 0),
    }
    del model
    torch.cuda.empty_cache()

# Save
out = Path("outputs/metrics/inference_time.json")
json.dump(results, open(out, "w"), indent=2)
print(f"\nSaved: {out}")

# LaTeX snippet
print("\nLaTeX rows (add to Tab. 1):")
for name, r in results.items():
    print(f"  {name} & {r['params_M']:.2f}M & {r['latency_bs1_ms']:.1f} ms & {r['throughput_wps']:,.0f} win/s \\\\")
