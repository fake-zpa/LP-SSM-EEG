# LP-SSM-EEG

**Locally-Predictive State Space Models with EEG-Specific Objectives for Cross-Patient Seizure Detection**

Pingan Zhang, Qin Zhou, Jiaqi Chen, Hui Zhang*

School of Mathematical Sciences, China West Normal University, Nanchong, Sichuan, China

> *Corresponding author: yiyeshenzhou@cwnu.edu.cn*

---

## Overview

LP-SSM-EEG augments a Mamba-based state-space backbone with **per-block auxiliary denoising objectives** and an **EEG spectral modulator** for cross-patient EEG seizure detection. Key components:

- **Local Denoising Head** — time-frequency (STFT) reconstruction + band-selective (δ,θ,α,β) reconstruction targets at every Mamba block, discarded at inference.
- **EEG Spectral Modulator** — computes a scalar weight *w* ∈ [0.5, 2.0] from band powers, ictal ratio, temporal variance, and uncertainty to reweight the auxiliary loss per window.
- **Warmup Schedule** — linearly ramps auxiliary loss coefficient λ from 0 to target over configurable epochs, reducing cross-seed variance by 3×.

All auxiliary heads are **discarded at inference**, so the runtime cost is identical to the Mamba baseline.

## Key Results

| Setting | LP-SSM-EEG | Mamba Baseline |
|---------|-----------|---------------|
| 8-seed AUROC (CHB-MIT) | **0.902 ± 0.017** | 0.885 ± 0.026 |
| Denoising head ablation | +0.048 AUROC | — |
| SIENA zero-shot AUPRC | **0.044 (+38%)** | 0.032 |
| Cross-seed variance | **0.017** | 0.026 |

## Project Structure

```
LP-SSM-EEG/
├── configs/              # YAML experiment configurations
│   ├── model/            # Model architectures (LP-SSM, Mamba, baselines)
│   ├── data/             # Dataset configs
│   └── train/            # Training hyperparameters
├── src/                  # Core source code
│   ├── cli/main.py       # Unified CLI (train / evaluate / preprocess)
│   ├── models/           # All model implementations
│   │   ├── lp_ssm_eeg.py       # LP-SSM-EEG model
│   │   ├── mamba_baseline.py    # Mamba baseline
│   │   ├── denoising_head.py    # TF + band-selective heads
│   │   ├── local_modulator.py   # EEG spectral modulator
│   │   └── losses.py            # Focal loss + local losses
│   ├── data/             # Dataset loading & preprocessing
│   ├── train/            # Trainer, early stopping, schedulers
│   ├── eval/             # Metrics, bootstrap CI, inference
│   └── utils/            # Seed, config, logging utilities
├── scripts/              # Analysis & plotting scripts
├── requirements.txt      # Python dependencies
├── environment.yml       # Conda environment
└── Makefile              # Convenience targets
```

## Quick Start

### 1. Environment Setup

```bash
# Option A: Conda (recommended)
conda env create -f environment.yml
conda activate mamba2

# Option B: pip
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.1+, CUDA 11.8+

### 2. Download & Preprocess CHB-MIT

```bash
# Download targeted subjects (chb01-06, chb10)
python -m src.cli.main preprocess \
    --dataset chbmit \
    --subjects chb01 chb02 chb03 chb04 chb05 chb06 chb10

# Build train/val/test manifest
python -m src.cli.main build-manifest \
    --config configs/data/chbmit_primary.yaml
```

### 3. Train LP-SSM-EEG

```bash
python -m src.cli.main train \
    --config configs/model/lp_ssm_eeg.yaml \
    --seed 42
```

### 4. Train Mamba Baseline

```bash
python -m src.cli.main train \
    --config configs/model/mamba_baseline.yaml \
    --seed 42
```

### 5. Evaluate

```bash
# Evaluate with bootstrap 95% CI
python -m src.cli.main evaluate \
    --checkpoint outputs/checkpoints/<run_id>/best.pt \
    --config configs/model/lp_ssm_eeg.yaml
```

## Demo: Reproduce Main Result

```bash
# Full pipeline: download → preprocess → train both models → evaluate
make smoke-test          # Quick sanity check (1 subject, 2 epochs)
make train-all           # Full training (requires GPU, ~2h on RTX 4090)
```

## Datasets

| Dataset | Description | Link |
|---------|------------|------|
| CHB-MIT | Scalp EEG, 22 subjects, seizure detection | [PhysioNet](https://physionet.org/content/chbmit/1.0.0/) |
| SIENA | Scalp EEG, 14 subjects (zero-shot evaluation) | [PhysioNet](https://physionet.org/content/siena-scalp-eeg/1.0.0/) |

## Citation

If you find this work useful, please cite:

```bibtex
@article{zhang2026lpssmeeg,
  title={LP-SSM-EEG: Locally-Predictive State Space Models with EEG-Specific Objectives for Cross-Patient Seizure Detection},
  author={Zhang, Pingan and Zhou, Qin and Chen, Jiaqi and Zhang, Hui},
  journal={Biomedical Signal Processing and Control},
  year={2026}
}
```

## Acknowledgements

This work was supported by the National Natural Science Foundation of China under grant 12301525.

## License

This project is released for academic research purposes.
