"""Collect system/GPU/environment information for reproducibility logs."""
import sys
import platform
import datetime
from pathlib import Path
from typing import Dict, Any


def collect_system_info() -> Dict[str, Any]:
    info = {
        "timestamp": datetime.datetime.now().isoformat(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }

    try:
        import psutil
        info["cpu_count_logical"] = psutil.cpu_count(logical=True)
        info["cpu_count_physical"] = psutil.cpu_count(logical=False)
        mem = psutil.virtual_memory()
        info["ram_total_gb"] = round(mem.total / 1e9, 2)
        info["ram_available_gb"] = round(mem.available / 1e9, 2)
    except ImportError:
        info["psutil"] = "not installed"

    try:
        import torch
        info["torch_version"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda
            info["cudnn_version"] = torch.backends.cudnn.version()
            info["gpu_count"] = torch.cuda.device_count()
            info["gpus"] = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                info["gpus"].append({
                    "index": i,
                    "name": props.name,
                    "vram_total_gb": round(props.total_memory / 1e9, 2),
                    "multi_processor_count": props.multi_processor_count,
                })
    except ImportError:
        info["torch"] = "not installed"

    return info


def print_system_info(info: Dict[str, Any]) -> None:
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    for k, v in info.items():
        if k == "gpus":
            for gpu in v:
                print(f"  GPU[{gpu['index']}]: {gpu['name']} | {gpu['vram_total_gb']} GB VRAM")
        else:
            print(f"  {k}: {v}")
    print("=" * 60)


def save_system_info(info: Dict[str, Any], log_dir: str) -> str:
    import json
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(log_dir) / f"system_info_{ts}.json"
    with open(path, "w") as f:
        json.dump(info, f, indent=2)
    return str(path)


def format_system_info_md(info: Dict[str, Any]) -> str:
    lines = ["## System Information\n"]
    lines.append(f"- **Timestamp**: {info.get('timestamp', 'N/A')}")
    lines.append(f"- **Python**: {info.get('python_version', 'N/A').split()[0]}")
    lines.append(f"- **Platform**: {info.get('platform', 'N/A')}")
    lines.append(f"- **PyTorch**: {info.get('torch_version', 'N/A')}")
    lines.append(f"- **CUDA Available**: {info.get('cuda_available', False)}")
    lines.append(f"- **CUDA Version**: {info.get('cuda_version', 'N/A')}")
    lines.append(f"- **RAM Total**: {info.get('ram_total_gb', 'N/A')} GB")
    for gpu in info.get("gpus", []):
        lines.append(f"- **GPU[{gpu['index']}]**: {gpu['name']} | {gpu['vram_total_gb']} GB VRAM")
    return "\n".join(lines)
