"""Patient sequences: cached region-graph node features per visit + days + clinical vector + labels."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.utils import normalize_pid


def prepare_clinical(cfg, train_pids):
    """Returns (features_df, mask_df, labels_df) indexed by patient id (UCSD-PTGBM-XXXX)."""
    c = cfg["clinical"]
    df = pd.read_csv(cfg["paths"]["clinical_csv"])
    needed = [c["id_col"], c["os_time_col"], c["os_event_col"], c["progression_col"]] \
        + list(c["numeric_cols"]) + list(c["categorical_cols"])
    missing = [n for n in needed if n not in df.columns]
    if missing:
        raise KeyError(f"Columns {missing} not found in clinical CSV.\n"
                       f"Available columns: {list(df.columns)}\nEdit the 'clinical:' section of the config.")
    df["pid"] = df[c["id_col"]].map(normalize_pid)
    df = df.dropna(subset=["pid"]).drop_duplicates("pid").set_index("pid")

    feats, masks = [], []
    if c["numeric_cols"]:
        num = df[c["numeric_cols"]].apply(pd.to_numeric, errors="coerce")
        tr = num.loc[num.index.isin(train_pids)]
        num = (num - tr.mean()) / tr.std().replace(0, 1).fillna(1)
        feats.append(num.fillna(0.0))
        masks.append(num.notna().astype(float))
    for col in c["categorical_cols"]:
        s = df[col].where(df[col].notna())
        d = pd.get_dummies(s, prefix=col).astype(float)
        feats.append(d)
        masks.append(pd.DataFrame(np.repeat(s.notna().values[:, None].astype(float), d.shape[1], 1),
                                  index=df.index, columns=d.columns))
    if feats:
        X, M = pd.concat(feats, axis=1), pd.concat(masks, axis=1)
    else:
        X, M = pd.DataFrame(index=df.index), pd.DataFrame(index=df.index)

    lab = pd.DataFrame({
        "os_time": pd.to_numeric(df[c["os_time_col"]], errors="coerce"),
        "os_event": pd.to_numeric(df[c["os_event_col"]], errors="coerce"),
        "prog": pd.to_numeric(df[c["progression_col"]], errors="coerce"),
    }, index=df.index)
    return X, M, lab


def build_samples(cfg, scan_df, X, M, lab):
    L = cfg["longitudinal"]
    feat_dir = Path(cfg["paths"]["feature_dir"])
    visit_days = None
    if cfg["paths"].get("visits_csv"):
        v = pd.read_csv(cfg["paths"]["visits_csv"])
        v["pid"] = v["patient_id"].map(normalize_pid)
        visit_days = {(r.pid, int(r.visit)): float(r.days) for r in v.itertuples()}
    else:
        print(f"[warn] visits_csv not set: assuming {L['default_interval_days']} days between visits.")

    samples, skipped = {}, 0
    for pid, g in scan_df.groupby("patient_id"):
        if pid not in lab.index or lab.loc[pid].isna().any():
            skipped += 1
            continue
        g = g.sort_values("visit")
        if L.get("max_visits"):
            g = g.head(int(L["max_visits"]))
        nodes, days = [], []
        for r in g.itertuples():
            f = feat_dir / f"{pid}_{int(r.visit):02d}.pt"
            if not f.exists():
                continue
            nodes.append(torch.load(f)["nodes"])
            if visit_days is not None:
                days.append(visit_days.get((pid, int(r.visit)), np.nan))
            else:
                days.append((int(r.visit) - 1) * float(L["default_interval_days"]))
        if not nodes or np.isnan(days).any():
            skipped += 1
            continue
        samples[pid] = {
            "nodes": torch.stack(nodes), "times": torch.tensor(days, dtype=torch.float32),
            "clin": torch.tensor(X.loc[pid].values, dtype=torch.float32),
            "clin_mask": torch.tensor(M.loc[pid].values, dtype=torch.float32),
            "prog": float(lab.loc[pid, "prog"]), "os_time": float(lab.loc[pid, "os_time"]),
            "os_event": float(lab.loc[pid, "os_event"]), "pid": pid,
        }
    print(f"Usable patients: {len(samples)} (skipped {skipped}: missing labels / features / visit days)")
    return samples


class LongitudinalDataset(Dataset):
    def __init__(self, samples, pids):
        self.items = [samples[p] for p in pids if p in samples]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate(batch):
    B = len(batch)
    T = max(b["nodes"].shape[0] for b in batch)
    N, Fd = batch[0]["nodes"].shape[1:]
    nodes = torch.zeros(B, T, N, Fd)
    times = torch.zeros(B, T)
    valid = torch.zeros(B, T, dtype=torch.bool)
    for i, b in enumerate(batch):
        t = b["nodes"].shape[0]
        nodes[i, :t], times[i, :t], valid[i, :t] = b["nodes"], b["times"], True
    return {
        "nodes": nodes, "times": times, "valid": valid,
        "clin": torch.stack([b["clin"] for b in batch]),
        "clin_mask": torch.stack([b["clin_mask"] for b in batch]),
        "prog": torch.tensor([b["prog"] for b in batch]),
        "os_time": torch.tensor([b["os_time"] for b in batch]),
        "os_event": torch.tensor([b["os_event"] for b in batch]),
        "pid": [b["pid"] for b in batch],
    }
