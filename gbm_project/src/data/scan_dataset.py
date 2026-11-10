"""
Step 1: walk the UCSD-PTGBM folders and build a table (one row per patient visit).

    python -m src.data.scan_dataset --config configs/config.yaml
"""
import argparse
import re
from pathlib import Path

import pandas as pd

from src.utils import load_config

FOLDER_RE = re.compile(r"(UCSD-PTGBM-\d+)_(\d+)", re.I)

# Order matters: the first rule that matches wins.
RULES = [
    ("seg",   ("seg", "mask", "label", "tumor")),
    ("flair", ("flair",)),
    ("t1c",   ("t1post", "t1c", "t1ce", "t1gd", "t1_post", "post", "contrast")),
    ("t2",    ("t2",)),
    ("t1",    ("t1",)),
]


def classify(filename):
    n = filename.lower()
    for key, words in RULES:
        if any(w in n for w in words):
            return key
    return None


def scan(root):
    rows = []
    for folder in sorted(Path(root).iterdir()):
        m = FOLDER_RE.fullmatch(folder.name)
        if not (folder.is_dir() and m):
            continue
        found = {k: [] for k, _ in RULES}
        for f in sorted(folder.rglob("*")):
            if f.is_file() and f.name.lower().endswith((".nii", ".nii.gz")):
                key = classify(f.name)
                if key:
                    found[key].append(str(f))
        row = {"patient_id": m.group(1).upper(), "visit": int(m.group(2)), "folder": str(folder)}
        for key, paths in found.items():
            if len(paths) > 1:
                print(f"[warn] {folder.name}: {len(paths)} candidates for '{key}', using {Path(paths[0]).name}")
            row[key] = paths[0] if paths else ""
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(df):
    print(f"\nPatients: {df.patient_id.nunique()} | visits: {len(df)}")
    print("Visits per patient:\n", df.groupby("patient_id").size().value_counts().sort_index().to_string())
    for k in ("flair", "t1", "t1c", "t2", "seg"):
        print(f"  {k:6s} present in {(df[k] != '').sum()} / {len(df)} visits")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    df = scan(cfg["paths"]["data_root"])
    if df.empty:
        raise SystemExit("No UCSD-PTGBM-XXXX_YY folders found. Check paths.data_root in the config.")
    out = Path(cfg["paths"]["scan_table"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    summarize(df)
    print(f"\nSaved -> {out}")
    print("Open the CSV and check that each modality column points to the right file.")


if __name__ == "__main__":
    main()
