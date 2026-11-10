"""Segmentation dataset: loads the 4 MRI sequences (+ mask) of one visit as a fixed-size 3D tensor."""
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from src.data.preprocess import brain_bbox, labels_to_regions, zscore


def load_volume(path):
    img = nib.load(str(path))
    v = np.asarray(img.dataobj, dtype=np.float32)
    if v.ndim == 4:
        v = v[..., 0]
    return v, img.header.get_zooms()[:3]


class SegDataset(Dataset):
    def __init__(self, df, cfg, train=False, need_seg=True):
        self.rows = df.reset_index(drop=True)
        self.mods = cfg["data"]["modalities"]
        self.size = tuple(cfg["data"]["spatial_size"])
        self.data_cfg = cfg["data"]
        self.train = train
        self.need_seg = need_seg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        vols, zooms = [], (1.0, 1.0, 1.0)
        for m in self.mods:
            p = r.get(m, "")
            if isinstance(p, str) and p:
                v, zooms = load_volume(p)
                vols.append(zscore(v))
            else:
                vols.append(None)
        ref = next(v for v in vols if v is not None)
        x = np.stack([v if v is not None else np.zeros_like(ref) for v in vols])   # (C,X,Y,Z)

        seg = None
        if self.need_seg:
            seg_v, _ = load_volume(r["seg"])
            seg = np.rint(seg_v).astype(np.int16)

        sl = brain_bbox(x)
        x = x[(slice(None),) + sl]
        crop_shape = np.array(x.shape[1:], dtype=np.float32)
        spacing = np.array(zooms, dtype=np.float32) * crop_shape / np.array(self.size, dtype=np.float32)

        x = F.interpolate(torch.from_numpy(np.ascontiguousarray(x))[None], size=self.size,
                          mode="trilinear", align_corners=False)[0]
        out = {"id": f"{r['patient_id']}_{int(r['visit']):02d}",
               "spacing": torch.from_numpy(spacing),
               "voxel_vol_mm3": torch.tensor(float(np.prod(spacing)))}

        mask = None
        if seg is not None:
            m = labels_to_regions(seg[sl], self.data_cfg)                  # (3,X,Y,Z)
            mask = F.interpolate(torch.from_numpy(np.ascontiguousarray(m))[None], size=self.size,
                                 mode="nearest")[0]
            mask = (mask > 0.5).float()

        if self.train:
            for ax in (1, 2, 3):
                if torch.rand(1).item() < 0.5:
                    x = torch.flip(x, dims=[ax])
                    if mask is not None:
                        mask = torch.flip(mask, dims=[ax])
            x = x * (0.9 + 0.2 * torch.rand(1).item()) + (torch.rand(1).item() - 0.5) * 0.2

        out["image"] = x.float()
        if mask is not None:
            out["mask"] = mask
        return out
