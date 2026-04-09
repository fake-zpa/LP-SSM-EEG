"""File I/O utilities."""
import json
import csv
import pickle
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


def save_json(obj: Any, path: Union[str, Path], indent: int = 2) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=str)


def load_json(path: Union[str, Path]) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def append_jsonl(obj: Any, path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(obj, default=str) + "\n")


def save_csv(rows: List[Dict], path: Union[str, Path], fieldnames: Optional[List[str]] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_csv(path: Union[str, Path]) -> List[Dict]:
    with open(path, "r") as f:
        return list(csv.DictReader(f))


def save_pickle(obj: Any, path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(path: Union[str, Path]) -> Any:
    with open(path, "rb") as f:
        return pickle.load(f)


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p
