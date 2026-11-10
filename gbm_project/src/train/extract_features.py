"""
Step 3: run the trained segmentation network on EVERY visit and cache the 4 region-graph nodes
(ET / necrotic core / edema / whole tumour) so the longitudinal model trains fast.

    python -m src.train.extract_features --config configs/config.yaml
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.seg_dataset import SegDataset
from src.models.region_nodes import region_nodes
from src.models.seg_swin_cnn import HybridCNNSwinSeg
from src.utils import load_config


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = HybridCNNSwinSeg(in_ch=len(cfg["data"]["modalities"]), base=cfg["seg"]["base_channels"]).to(device)
    model.load_state_dict(torch.load(cfg["paths"]["seg_ckpt"], map_location=device)["model"])
    model.eval()

    df = pd.read_csv(cfg["paths"]["scan_table"]).fillna("")
    out_dir = Path(cfg["paths"]["feature_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    loader = DataLoader(SegDataset(df, cfg, train=False, need_seg=False), batch_size=1, num_workers=2)

    for batch in tqdm(loader, desc="extracting"):
        logits, bott = model(batch["image"].to(device), return_feats=True)
        nodes = region_nodes(logits, bott, batch["voxel_vol_mm3"].to(device))[0].cpu()
        torch.save({"nodes": nodes}, out_dir / f"{batch['id'][0]}.pt")
    print(f"Saved node features for {len(df)} visits -> {out_dir}")


if __name__ == "__main__":
    main()
