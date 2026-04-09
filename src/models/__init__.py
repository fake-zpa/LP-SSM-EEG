"""Model registry."""
from src.models.eegnet import EEGNet
from src.models.cnn_baseline import CNNBaseline
from src.models.transformer_baseline import TransformerBaseline
from src.models.mamba_baseline import MambaBaseline
from src.models.lp_ssm_eeg import LPSSMEEG

MODEL_REGISTRY = {
    "eegnet": EEGNet,
    "cnn_baseline": CNNBaseline,
    "transformer_baseline": TransformerBaseline,
    "mamba_baseline": MambaBaseline,
    "mamba_large": MambaBaseline,
    "lp_ssm_eeg": LPSSMEEG,
}


def build_model(name: str, **kwargs):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](**kwargs)
