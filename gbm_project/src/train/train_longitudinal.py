"""
Step 4: GAT + time-aware Transformer + gated fusion, trained jointly for
progression classification (BCE) and overall-survival prediction (Cox partial likelihood).

    python -m src.train.train_longitudinal --config configs/config.yaml
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader

from src.data.longitudinal_dataset import LongitudinalDataset, build_samples, collate, prepare_clinical
from src.losses.survival_loss import cox_ph_loss
from src.metrics.survival import concordance_index, evaluate_survival
from src.models.full_model import LongitudinalGBMModel
from src.utils import load_config, load_or_make_splits, save_json, set_seed


def to_dev(b, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}


def forward(model, b):
    return model(b["nodes"], b["times"], b["valid"], b["clin"], b["clin_mask"])


@torch.no_grad()
def predict(model, loader, dev):
    model.eval()
    P, R, Y, T, E = [], [], [], [], []
    for b in loader:
        b = to_dev(b, dev)
        o = forward(model, b)
        P.append(torch.sigmoid(o["prog_logit"]).cpu()); R.append(o["risk"].cpu())
        Y.append(b["prog"].cpu()); T.append(b["os_time"].cpu()); E.append(b["os_event"].cpu())
    cat = lambda x: torch.cat(x).numpy()
    return cat(P), cat(R), cat(Y), cat(T), cat(E)


def prog_metrics(p, y):
    out = {"auroc": float("nan"), "auprc": float("nan"), "f1": float("nan")}
    if len(np.unique(y)) == 2:
        out["auroc"] = float(roc_auc_score(y, p))
        out["auprc"] = float(average_precision_score(y, p))
    out["f1"] = float(f1_score(y, (p > 0.5).astype(int), zero_division=0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/config.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    L = cfg["longitudinal"]
    set_seed(cfg["seed"])
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    scan = pd.read_csv(cfg["paths"]["scan_table"]).fillna("")
    splits = load_or_make_splits(cfg, sorted(scan.patient_id.unique()))
    X, M, lab = prepare_clinical(cfg, set(splits["train"]))
    samples = build_samples(cfg, scan, X, M, lab)

    dl = lambda key, shuffle: DataLoader(LongitudinalDataset(samples, splits[key]), batch_size=L["batch_size"],
                                         shuffle=shuffle, collate_fn=collate)
    tr, tr_eval, va, te = dl("train", True), dl("train", False), dl("val", False), dl("test", False)
    print({k: len(d.dataset) for k, d in zip(["train", "val", "test"], [tr, va, te])})

    first = next(iter(samples.values()))
    model = LongitudinalGBMModel(first["nodes"].shape[-1], first["clin"].shape[0], L["d_model"], L["gat_heads"],
                                 L["gat_layers"], L["temporal_heads"], L["temporal_layers"], L["dropout"]).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=L["lr"], weight_decay=L["weight_decay"])

    ys = np.array([s["prog"] for p, s in samples.items() if p in set(splits["train"])])
    pos_w = torch.tensor((len(ys) - ys.sum()) / max(ys.sum(), 1), dtype=torch.float32, device=dev)

    ckpt = Path(cfg["paths"]["out_dir"]) / "longitudinal_best.pt"
    best = -1.0
    for ep in range(1, L["epochs"] + 1):
        model.train()
        tot = 0.0
        for b in tr:
            b = to_dev(b, dev)
            o = forward(model, b)
            l_prog = F.binary_cross_entropy_with_logits(o["prog_logit"], b["prog"], pos_weight=pos_w)
            l_surv = cox_ph_loss(o["risk"], b["os_time"], b["os_event"])
            loss = L["w_progression"] * l_prog + L["w_survival"] * l_surv
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        p, r, y, t, e = predict(model, va, dev)
        pm, ci = prog_metrics(p, y), concordance_index(t, e, r)
        score = np.nanmean([pm["auroc"], ci])
        if ep % 5 == 0 or ep == 1:
            print(f"ep {ep:3d} | loss {tot / max(len(tr), 1):.4f} | val AUROC {pm['auroc']:.3f} "
                  f"AUPRC {pm['auprc']:.3f} F1 {pm['f1']:.3f} | C-index {ci:.3f}")
        if not np.isnan(score) and score > best:
            best = score
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=dev))
    p, r, y, t, e = predict(model, te, dev)
    _, r_tr, _, t_tr, e_tr = predict(model, tr_eval, dev)
    results = {"progression": prog_metrics(p, y), "survival": evaluate_survival(t_tr, e_tr, r_tr, t, e, r)}
    print("\nTEST:", results)
    save_json(results, Path(cfg["paths"]["out_dir"]) / "longitudinal_test_metrics.json")


if __name__ == "__main__":
    main()
