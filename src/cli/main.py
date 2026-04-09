"""
LP-SSM-EEG CLI entry point.

Commands:
  system-info   -- print and log system/GPU information
  download      -- download PhysioNet datasets
  verify        -- verify downloaded files
  preprocess    -- preprocess raw EDF data
  train         -- train a model
  eval          -- evaluate a trained model
"""
import argparse
import logging
import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def cmd_system_info(args):
    from src.utils.system_info import collect_system_info, print_system_info, save_system_info, format_system_info_md
    from src.utils.logger import get_logger

    log_dir = args.log_dir or "logs/setup"
    logger = get_logger("system_info", log_dir=log_dir)

    info = collect_system_info()
    print_system_info(info)
    saved_path = save_system_info(info, log_dir)
    logger.info(f"System info saved to {saved_path}")

    # Update PROJECT_STATUS.md
    status_path = PROJECT_ROOT / "docs" / "PROJECT_STATUS.md"
    if status_path.exists():
        content = status_path.read_text()
        md_section = format_system_info_md(info)
        if "*(Will be filled" in content:
            content = content.replace(
                "*(Will be filled by `make check-env` / `python src/cli/main.py system-info`)*\n\n- **Timestamp**: TBD\n- **GPU**: TBD\n- **CUDA**: TBD\n- **PyTorch**: TBD",
                md_section,
            )
            status_path.write_text(content)
            logger.info("Updated docs/PROJECT_STATUS.md with system info")

    return info


def cmd_download(args):
    from src.data.download_physionet import download_dataset
    from src.utils.logger import get_logger

    log_dir = args.log_dir or f"logs/download"
    logger = get_logger("download", log_dir=log_dir)

    subjects = args.subjects.split(",") if args.subjects else None
    max_files = args.max_files

    logger.info(f"Downloading {args.dataset}, subjects={subjects}, max_files={max_files}")
    summary = download_dataset(
        dataset=args.dataset,
        project_root=str(PROJECT_ROOT),
        subjects=subjects,
        max_files=max_files,
        log_dir=log_dir,
    )

    logger.info(f"Summary: {summary}")
    print(json.dumps(summary, indent=2))
    return summary


def cmd_verify(args):
    from src.data.verify_downloads import verify_dataset
    from src.utils.logger import get_logger

    logger = get_logger("verify", log_dir="logs/download")
    subjects = args.subjects.split(",") if args.subjects else None
    result = verify_dataset(args.dataset, project_root=str(PROJECT_ROOT), subjects=subjects)
    print(json.dumps(result, indent=2))
    return result


def cmd_preprocess(args):
    from src.utils.logger import get_logger
    from src.utils.seed import set_seed

    logger = get_logger("preprocess", log_dir=args.log_dir or "logs/preprocess")
    set_seed(42)

    subjects = args.subjects.split(",") if args.subjects else None

    if args.dataset == "chbmit":
        from src.data.preprocess_chbmit import preprocess_chbmit
        manifest = preprocess_chbmit(
            raw_dir=str(PROJECT_ROOT / "data" / "raw" / "chbmit"),
            output_dir=str(PROJECT_ROOT / "data" / "processed" / "chbmit"),
            manifest_path=str(PROJECT_ROOT / "data" / "manifests" / "chbmit_manifest.csv"),
            subjects=subjects,
            smoke=args.smoke,
        )
        logger.info(f"CHB-MIT preprocessing done: {len(manifest)} windows")

    elif args.dataset == "sleepedf":
        from src.data.preprocess_sleepedf import preprocess_sleepedf
        manifest = preprocess_sleepedf(
            raw_dir=str(PROJECT_ROOT / "data" / "raw" / "sleepedf"),
            output_dir=str(PROJECT_ROOT / "data" / "processed" / "sleepedf"),
            manifest_path=str(PROJECT_ROOT / "data" / "manifests" / "sleepedf_manifest.csv"),
            subjects=subjects,
            smoke=args.smoke,
        )
        logger.info(f"Sleep-EDF preprocessing done: {len(manifest)} windows")
    else:
        logger.error(f"Unsupported dataset: {args.dataset}")


def cmd_train(args):
    import torch
    from src.utils.logger import get_logger
    from src.utils.seed import set_seed
    from src.utils.config import load_config
    from src.utils.reproducibility import generate_run_id, save_run_manifest
    from src.models import build_model
    from src.models.losses import LPSSMLoss
    from src.train.trainer import Trainer
    from src.train.amp import AMPContext
    from src.train.early_stopping import EarlyStopping
    from src.train.optim import build_optimizer
    from src.train.schedulers import build_scheduler
    from torch.utils.data import DataLoader

    cfg_path = args.config or str(PROJECT_ROOT / "configs" / "train" / "amp_single_4090.yaml")
    cfg = load_config(cfg_path, defaults_path=str(PROJECT_ROOT / "configs" / "default.yaml"))

    seed = getattr(args, "seed", None) or cfg.get("seed", 42)
    set_seed(seed)

    log_dir = args.log_dir or "logs/train"
    run_id = generate_run_id(prefix=f"{args.model}_")
    logger = get_logger(f"train_{args.model}", log_dir=f"{log_dir}/{run_id}")

    max_epochs = getattr(args, "max_epochs", None) or cfg.get("max_epochs", 100)
    training_mode = getattr(args, "training_mode", None) or "local"
    subjects = args.subjects.split(",") if args.subjects else None

    logger.info(f"Training model={args.model}, dataset={args.dataset}, mode={training_mode}, run_id={run_id}")

    # --- Build dataset ---
    manifest_map = {
        "chbmit": str(PROJECT_ROOT / "data" / "manifests" / "chbmit_manifest.csv"),
        "sleepedf": str(PROJECT_ROOT / "data" / "manifests" / "sleepedf_manifest.csv"),
    }
    # Allow custom manifest override (e.g., harder split)
    manifest_path = getattr(args, "manifest", None) or manifest_map.get(args.dataset)
    if not manifest_path or not Path(manifest_path).exists():
        logger.error(f"Manifest not found for {args.dataset}. Run preprocessing first.")
        return
    logger.info(f"Using manifest: {manifest_path}")

    if args.dataset == "chbmit":
        from src.data.dataset_chbmit import CHBMITDataset
        from src.data.transforms import get_train_transforms
        context_sec_val = getattr(args, "context_sec", None) or 4.0
        max_samples_val = int(context_sec_val * 256) if context_sec_val < 4.0 else None
        train_ds = CHBMITDataset(manifest_path, split="train", subjects=subjects, transform=get_train_transforms(), max_samples=max_samples_val)
        val_ds = CHBMITDataset(manifest_path, split="val", subjects=subjects, max_samples=max_samples_val)
        n_classes = 2
        # Auto-detect channel count from manifest (may vary by preprocessing run)
        import pandas as _pd
        _mdf = _pd.read_csv(manifest_path)
        in_channels = int(_mdf["n_channels"].iloc[0]) if "n_channels" in _mdf.columns else 23
        class_weights = train_ds.class_weights().to("cuda" if torch.cuda.is_available() else "cpu")
    elif args.dataset == "sleepedf":
        from src.data.dataset_sleepedf import SleepEDFDataset
        train_ds = SleepEDFDataset(manifest_path, split="train", subjects=subjects)
        val_ds = SleepEDFDataset(manifest_path, split="val", subjects=subjects)
        n_classes = 5
        in_channels = 2
        class_weights = None
    else:
        logger.error(f"Unknown dataset: {args.dataset}")
        return

    if len(train_ds) == 0 or len(val_ds) == 0:
        logger.error("Empty dataset split. Check preprocessing.")
        return

    batch_size = cfg.get("batch_size", 512)
    smoke_mode = getattr(args, "smoke", False)
    if smoke_mode:
        max_epochs = min(max_epochs, 3)
        batch_size = min(batch_size, 8)

    num_workers = cfg.get("num_workers", 8)
    persistent_workers = cfg.get("persistent_workers", True) and num_workers > 0
    prefetch_factor = cfg.get("prefetch_factor", 4) if num_workers > 0 else None
    dl_kwargs = dict(
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=persistent_workers,
    )
    if prefetch_factor is not None:
        dl_kwargs["prefetch_factor"] = prefetch_factor

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, **dl_kwargs)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, **dl_kwargs)

    # --- Build model ---
    context_sec = getattr(args, "context_sec", None) or 4.0
    window_samples = int(context_sec * 256)

    # Load architecture params from model YAML config
    model_cfg_path = PROJECT_ROOT / "configs" / "model" / f"{args.model}.yaml"
    model_arch = {}
    if model_cfg_path.exists():
        _mcfg = load_config(str(model_cfg_path))
        model_arch = _mcfg.get("architecture", {})

    model_kwargs = dict(in_channels=in_channels, n_classes=n_classes)
    # Apply architecture overrides from model YAML (d_model, n_layers, etc.)
    for _k in ("d_model", "d_state", "d_conv", "expand", "n_layers", "dropout"):
        if _k in model_arch:
            model_kwargs[_k] = model_arch[_k]
    if getattr(args, "dropout", None) is not None:
        model_kwargs["dropout"] = args.dropout

    if args.model == "eegnet":
        model_kwargs.update(window_samples=window_samples)
    elif args.model == "lp_ssm_eeg":
        model_kwargs["training_mode"] = training_mode
        _use_modulator = not getattr(args, "no_modulator", False)
        model_kwargs["use_modulator"] = _use_modulator
        # Modulator feature ablation: pass through to build_model -> LPSSMEEG
        mod_features_str = getattr(args, "mod_features", None) or "band_powers,ictal_ratio,temporal_var,event_uncertainty"
        mod_features = set(f.strip() for f in mod_features_str.split(","))
        model_kwargs["mod_use_band_powers"] = "band_powers" in mod_features
        model_kwargs["mod_use_ictal_ratio"] = "ictal_ratio" in mod_features
        model_kwargs["mod_use_temporal_variance"] = "temporal_var" in mod_features
        model_kwargs["mod_use_event_uncertainty"] = "event_uncertainty" in mod_features
        # E06: global+modulator — ictal ratio weights the main task loss (no local objectives)
        if training_mode == "global" and _use_modulator and "ictal_ratio" in mod_features:
            model_kwargs["use_main_loss_mod"] = True
        # Denoising head ablation flags
        if getattr(args, "no_band_loss", False):
            model_kwargs["temporal_consistency_enabled"] = False
        if getattr(args, "no_tf_loss", False):
            model_kwargs["tf_reconstruction_enabled"] = False

    # Smoke mode: reduce model size only if mamba-ssm CUDA kernel is unavailable
    # When CUDA kernel is available, full model (d=256) fits in <2 GB VRAM
    if smoke_mode and args.model in ["mamba_baseline", "lp_ssm_eeg"]:
        try:
            from src.models.mamba_baseline import MAMBA_SSM_AVAILABLE
        except ImportError:
            MAMBA_SSM_AVAILABLE = False
        if not MAMBA_SSM_AVAILABLE:
            model_kwargs.update(d_model=64, d_state=8, n_layers=2, expand=1)
            logger.info("Smoke mode: reduced model size (d_model=64, d_state=8, n_layers=2) for VRAM budget (pure-PyTorch scan)")

    # cudnn auto-tuner: best kernel for fixed input size
    torch.backends.cudnn.benchmark = True

    model = build_model(args.model, **model_kwargs)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model {args.model}: {n_params/1e6:.2f}M parameters")

    # torch.compile for ~1.5-2x kernel fusion speedup (PyTorch 2.x)
    # suppress dynamo graph-break warnings for mamba_ssm CUDA custom ops
    import logging as _logging
    _logging.getLogger("torch._dynamo").setLevel(_logging.ERROR)
    try:
        model = torch.compile(model, mode="default")
        logger.info("torch.compile enabled (default mode)")
    except Exception as _e:
        logger.info(f"torch.compile skipped: {_e}")

    # --- Build loss ---
    _llw = getattr(args, "local_loss_weight", None)
    _llw = float(_llw) if _llw is not None else cfg.get("local_loss_weight", 0.15)
    loss_fn = LPSSMLoss(
        n_classes=n_classes,
        class_weights=class_weights,
        training_mode=training_mode if args.model == "lp_ssm_eeg" else "global",
        local_loss_weight=_llw,
    )

    # --- Build optimizer + scheduler ---
    opt_cfg = cfg.get("optimizer", {"name": "adamw", "lr": 5e-4, "weight_decay": 1e-4})
    if getattr(args, "lr", None) is not None:
        opt_cfg["lr"] = args.lr
    optimizer = build_optimizer(model, opt_cfg)

    sched_cfg = cfg.get("scheduler", {"name": "cosine"})
    scheduler = build_scheduler(optimizer, sched_cfg, steps_per_epoch=len(train_loader), max_epochs=max_epochs)

    amp_enabled = cfg.get("amp", True) and torch.cuda.is_available()
    amp_ctx = AMPContext(enabled=amp_enabled)

    _warmup = getattr(args, "warmup_epochs", None)
    _warmup = int(_warmup) if _warmup is not None else cfg.get("local_loss_warmup_epochs", 0)

    _min_es = getattr(args, "min_early_stop_epochs", None)
    _min_es = int(_min_es) if _min_es is not None else _warmup

    es_cfg = cfg.get("early_stopping", {})
    early_stopping = EarlyStopping(
        patience=es_cfg.get("patience", 15),
        mode=es_cfg.get("mode", "max"),
        min_delta=es_cfg.get("min_delta", 0.001),
        min_epochs=_min_es,
    ) if es_cfg.get("enabled", True) else None

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        amp_ctx=amp_ctx,
        max_epochs=max_epochs,
        gradient_accumulation_steps=cfg.get("gradient_accumulation_steps", 1),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),
        local_loss_warmup_epochs=_warmup,
        early_stopping=early_stopping,
        checkpoint_dir=str(PROJECT_ROOT / "outputs" / "checkpoints"),
        log_dir=log_dir,
        metrics_dir=str(PROJECT_ROOT / "outputs" / "metrics"),
        run_id=run_id,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    summary = trainer.fit(train_loader, val_loader)
    logger.info(f"Training complete: {summary}")
    return summary


def cmd_evaluate(args):
    """Evaluate a saved checkpoint on the test split."""
    import json
    import torch
    from torch.utils.data import DataLoader
    from src.models import build_model

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s][%(levelname)s][%(name)s] %(message)s")
    logger = logging.getLogger(f"eval_{args.model}")

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        logger.error(f"Checkpoint not found: {ckpt_path}")
        return

    manifest_map = {
        "chbmit": str(PROJECT_ROOT / "data" / "manifests" / "chbmit_manifest.csv"),
    }
    manifest_path = getattr(args, "manifest", None) or manifest_map.get(args.dataset)
    if not manifest_path or not Path(manifest_path).exists():
        logger.error(f"Manifest not found for {args.dataset}")
        return

    import pandas as _pd
    _mdf = _pd.read_csv(manifest_path)
    in_channels = int(_mdf["n_channels"].iloc[0]) if "n_channels" in _mdf.columns else 22
    n_classes = 2

    # Load architecture params from model YAML config (matches training)
    from src.utils.config import load_config as _load_config
    _model_cfg_path = PROJECT_ROOT / "configs" / "model" / f"{args.model}.yaml"
    _model_arch = {}
    if _model_cfg_path.exists():
        _mcfg = _load_config(str(_model_cfg_path))
        _model_arch = _mcfg.get("architecture", {})

    model_kwargs = dict(in_channels=in_channels, n_classes=n_classes)
    for _k in ("d_model", "d_state", "d_conv", "expand", "n_layers", "dropout"):
        if _k in _model_arch:
            model_kwargs[_k] = _model_arch[_k]
    if args.model == "lp_ssm_eeg":
        model_kwargs["training_mode"] = getattr(args, "training_mode", "global")
        if getattr(args, "no_band_loss", False):
            model_kwargs["temporal_consistency_enabled"] = False
        if getattr(args, "no_tf_loss", False):
            model_kwargs["tf_reconstruction_enabled"] = False
        _mod_features = getattr(args, "mod_features", None)
        if _mod_features:
            _feats = set(f.strip() for f in _mod_features.split(","))
            model_kwargs["mod_use_band_powers"] = "band_powers" in _feats
            model_kwargs["mod_use_ictal_ratio"] = "ictal_ratio" in _feats
            model_kwargs["mod_use_temporal_variance"] = "temporal_var" in _feats
            model_kwargs["mod_use_event_uncertainty"] = "event_uncertainty" in _feats
        if getattr(args, "no_modulator", False):
            # Keep full modulator architecture (feature flags stay True) so the
            # checkpoint state_dict loads correctly; just disable the output path.
            model_kwargs["use_modulator"] = False
    elif args.model == "eegnet":
        model_kwargs["window_samples"] = 1024
    model = build_model(args.model, **model_kwargs)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt["model_state"]
    # Handle torch.compile wrapper: strip _orig_mod. prefix if present
    if any(k.startswith("_orig_mod.") for k in state):
        state = {k.replace("_orig_mod.", "", 1): v for k, v in state.items()}
    model.load_state_dict(state)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    from src.data.dataset_chbmit import CHBMITDataset
    test_ds = CHBMITDataset(manifest_path, split="test")
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)
    logger.info(f"Test set: {len(test_ds)} windows")

    from src.eval.metrics_classification import compute_metrics
    all_logits, all_targets = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            out = model(x)
            logits = out["logits"] if isinstance(out, dict) else out
            all_logits.append(logits.float().cpu())
            all_targets.append(y)

    logits_cat = torch.cat(all_logits, dim=0)
    targets_cat = torch.cat(all_targets, dim=0)
    metrics = compute_metrics(logits_cat, targets_cat)
    logger.info(f"Test metrics [{args.model}]: {json.dumps({k: round(v, 4) for k, v in metrics.items()})}")
    print(json.dumps({"model": args.model, "run_id": ckpt.get("run_id"), **{k: round(v, 4) for k, v in metrics.items()}}, indent=2))
    return metrics


def main():
    parser = argparse.ArgumentParser(description="LP-SSM-EEG Research Project CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # system-info
    p_sysinfo = subparsers.add_parser("system-info", help="Print system/GPU info")
    p_sysinfo.add_argument("--log-dir", default="logs/setup")

    # download
    p_dl = subparsers.add_parser("download", help="Download PhysioNet dataset")
    p_dl.add_argument("--dataset", required=True, choices=["chbmit", "sleepedf", "eegmmidb"])
    p_dl.add_argument("--subjects", default=None, help="Comma-separated subject IDs, e.g. chb01,chb02")
    p_dl.add_argument("--max-files", type=int, default=None)
    p_dl.add_argument("--log-dir", default="logs/download")

    # verify
    p_ver = subparsers.add_parser("verify", help="Verify downloaded files")
    p_ver.add_argument("--dataset", required=True, choices=["chbmit", "sleepedf", "eegmmidb"])
    p_ver.add_argument("--subjects", default=None)

    # preprocess
    p_pp = subparsers.add_parser("preprocess", help="Preprocess raw EDF data")
    p_pp.add_argument("--dataset", required=True, choices=["chbmit", "sleepedf"])
    p_pp.add_argument("--subjects", default=None)
    p_pp.add_argument("--smoke", action="store_true", help="Process only 1-2 subjects")
    p_pp.add_argument("--log-dir", default="logs/preprocess")

    # evaluate
    p_ev = subparsers.add_parser("evaluate", help="Evaluate saved checkpoint on test split")
    p_ev.add_argument("--model", required=True, choices=["eegnet", "cnn_baseline", "transformer_baseline", "mamba_baseline", "mamba_large", "lp_ssm_eeg"])
    p_ev.add_argument("--checkpoint", required=True, help="Path to best.pt checkpoint")
    p_ev.add_argument("--dataset", default="chbmit", choices=["chbmit"])
    p_ev.add_argument("--training-mode", default="global", choices=["local", "global"])
    p_ev.add_argument("--no-band-loss", action="store_true", help="Match checkpoint trained with --no-band-loss")
    p_ev.add_argument("--no-tf-loss", action="store_true", help="Match checkpoint trained with --no-tf-loss")
    p_ev.add_argument("--no-modulator", action="store_true", help="Match checkpoint trained with --no-modulator")
    p_ev.add_argument("--mod-features", default=None, help="Comma-sep modulator features: band_powers,ictal_ratio,temporal_var,event_uncertainty")
    p_ev.add_argument("--manifest", default=None, help="Override manifest CSV path")
    p_ev.add_argument("--log-dir", default="logs/eval")

    # train
    p_tr = subparsers.add_parser("train", help="Train a model")
    p_tr.add_argument("--model", required=True, choices=["eegnet", "cnn_baseline", "transformer_baseline", "mamba_baseline", "mamba_large", "lp_ssm_eeg"])
    p_tr.add_argument("--dataset", default="chbmit", choices=["chbmit", "sleepedf"])
    p_tr.add_argument("--config", default=None)
    p_tr.add_argument("--subjects", default=None)
    p_tr.add_argument("--max-epochs", type=int, default=None)
    p_tr.add_argument("--smoke", action="store_true")
    p_tr.add_argument("--training-mode", default="local", choices=["local", "global", "mixed"])
    p_tr.add_argument("--context-sec", type=float, default=4.0)
    p_tr.add_argument("--no-modulator", action="store_true", help="Ablation: disable EEG local modulator")
    p_tr.add_argument("--mod-features", default=None, help="Comma-sep modulator features to enable: band_powers,ictal_ratio,temporal_var,event_uncertainty (default: all)")
    p_tr.add_argument("--no-band-loss", action="store_true", help="Ablation: disable band-selective reconstruction loss (TF head only)")
    p_tr.add_argument("--no-tf-loss", action="store_true", help="Ablation: disable TF reconstruction loss (band head only)")
    p_tr.add_argument("--manifest", default=None, help="Override manifest CSV path (e.g., for harder split)")
    p_tr.add_argument("--seed", type=int, default=None, help="Random seed override")
    p_tr.add_argument("--local-loss-weight", type=float, default=None, dest="local_loss_weight",
                      help="Override local loss weight lambda (default: 0.15)")
    p_tr.add_argument("--warmup-epochs", type=int, default=None, dest="warmup_epochs",
                      help="Override local loss warmup epochs (default: 10)")
    p_tr.add_argument("--min-early-stop-epochs", type=int, default=None, dest="min_early_stop_epochs",
                      help="Minimum epochs before early stopping may trigger (default: same as warmup-epochs)")
    p_tr.add_argument("--lr", type=float, default=None, help="Override optimizer learning rate")
    p_tr.add_argument("--dropout", type=float, default=None, help="Override model dropout rate")
    p_tr.add_argument("--log-dir", default="logs/train")
    p_tr.add_argument("--override", action="append", default=[], help="Override config key=value")

    args = parser.parse_args()

    if args.command == "system-info":
        cmd_system_info(args)
    elif args.command == "download":
        cmd_download(args)
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "preprocess":
        cmd_preprocess(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
