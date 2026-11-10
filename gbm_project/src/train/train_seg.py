"""
Step 2: train the hybrid 3D CNN-Swin segmentation network.

    python -m src.train.train_seg --config configs/config.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.seg_dataset import SegDataset
from src.losses.seg_loss import DiceBCELoss
from src.metrics.segmentation import REGIONS, region_metrics
from src.models.seg_swin_cnn import HybridCNNSwinSeg
from src.utils import load_config, load_or_make_splits, save_json, set_seed


@torch.no_grad()
def evaluate(model, loader, device, with_hd=False):
    model.eval()
    per_region = {r: {"dice": [], "iou": [], "hd95": []} for r in REGIONS}
    for batch in loader:
        x = batch["image"].to(device)
        pred = (torch.sigmoid(model(x)) > 0.5).cpu().numpy().astype(bool)
        gt = batch["mask"].numpy().astype(bool)
        for b in range(pred.shape[0]):
            m = region_metrics(pred[b], gt[b], batch["spacing"][b].numpy(), with_hd)
            for r in REGIONS:
                for k, v in m[r].items():
                    per_region[r][k].append(v)
    return {r: {k: float(np.nanmean(v)) for k, v in d.items() if v} for r, d in per_region.items()}


def mean_dice(res):
    return float(np.mean([res[r]["dice"] for r in REGIONS]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["seed"])
    S = cfg["seg"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    df = pd.read_csv(cfg["paths"]["scan_table"]).fillna("")
    splits = load_or_make_splits(cfg, sorted(df.patient_id.unique()))
    df = df[df.seg != ""]
    sub = {k: df[df.patient_id.isin(v)] for k, v in splits.items()}
    print({k: f"{len(v)} visits" for k, v in sub.items()})

    mk = lambda d, train: DataLoader(SegDataset(d, cfg, train=train), batch_size=S["batch_size"] if train else 1,
                                     shuffle=train, num_workers=S["num_workers"], pin_memory=True)
    tr, va, te = mk(sub["train"], True), mk(sub["val"], False), mk(sub["test"], False)

    model = HybridCNNSwinSeg(in_ch=len(cfg["data"]["modalities"]), base=S["base_channels"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=S["lr"], weight_decay=S["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=S["epochs"])
    loss_fn = DiceBCELoss()
    use_amp = bool(S["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    out_dir = Path(cfg["paths"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    best = -1.0
    for ep in range(1, S["epochs"] + 1):
        model.train()
        run = 0.0
        for batch in tqdm(tr, desc=f"epoch {ep}/{S['epochs']}", leave=False):
            x, y = batch["image"].to(device), batch["mask"].to(device)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, enabled=use_amp):
                loss = loss_fn(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            run += loss.item()
        sched.step()
        res = evaluate(model, va, device)
        md = mean_dice(res)
        print(f"ep {ep:3d} | loss {run / max(len(tr), 1):.4f} | val Dice ET {res['ET']['dice']:.3f} "
              f"TC {res['TC']['dice']:.3f} WT {res['WT']['dice']:.3f} | mean {md:.3f}")
        if md > best:
            best = md
            torch.save({"model": model.state_dict(), "config": cfg}, cfg["paths"]["seg_ckpt"])

    ckpt = torch.load(cfg["paths"]["seg_ckpt"], map_location=device)
    model.load_state_dict(ckpt["model"])
    test_res = evaluate(model, te, device, with_hd=True)
    print("\nTEST (Dice / IoU / HD95 in mm):")
    for r in REGIONS:
        print(f"  {r}: {test_res[r]}")
    save_json(test_res, out_dir / "seg_test_metrics.json")


if __name__ == "__main__":
    main()
