import json
import random
import re
from pathlib import Path

import numpy as np
import yaml


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def normalize_pid(x):
    """'UCSD-PTGBM-0005', '0005', 5  ->  'UCSD-PTGBM-0005'"""
    m = re.findall(r"\d+", str(x))
    return f"UCSD-PTGBM-{int(m[0]):04d}" if m else None


def split_patients(pids, train=0.7, val=0.15, seed=42):
    """Patient-level split (never split visits of one patient across sets)."""
    pids = sorted(pids)
    rng = np.random.RandomState(seed)
    rng.shuffle(pids)
    n = len(pids)
    n_tr, n_va = int(n * train), int(n * val)
    return {"train": pids[:n_tr], "val": pids[n_tr:n_tr + n_va], "test": pids[n_tr + n_va:]}


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_or_make_splits(cfg, pids):
    p = Path(cfg["paths"]["out_dir"]) / "splits.json"
    if p.exists():
        return load_json(p)
    s = split_patients(pids, seed=cfg.get("seed", 42))
    save_json(s, p)
    return s
