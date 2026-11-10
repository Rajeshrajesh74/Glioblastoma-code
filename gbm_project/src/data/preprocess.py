"""Pure-numpy preprocessing helpers (no torch needed)."""
import numpy as np


def zscore(vol):
    """z-score using non-zero (brain) voxels only; background stays 0."""
    m = vol != 0
    if m.sum() < 10:
        return vol.astype(np.float32)
    mu, sd = vol[m].mean(), vol[m].std() + 1e-8
    out = (vol - mu) / sd
    out[~m] = 0
    return out.astype(np.float32)


def brain_bbox(stack, margin=4):
    """stack: (C,X,Y,Z). Returns tuple of 3 slices covering non-zero voxels."""
    nz = np.any(stack != 0, axis=0)
    if not nz.any():
        return tuple(slice(0, s) for s in nz.shape)
    sl = []
    for ax in range(3):
        other = tuple(a for a in range(3) if a != ax)
        idx = np.where(nz.any(axis=other))[0]
        sl.append(slice(max(int(idx.min()) - margin, 0), min(int(idx.max()) + 1 + margin, nz.shape[ax])))
    return tuple(sl)


def labels_to_regions(seg, data_cfg):
    """BraTS-style label map -> 3 binary channels [ET, TC, WT]."""
    et = np.isin(seg, data_cfg["et_labels"])
    tc = np.isin(seg, data_cfg["tc_labels"])
    wt = np.isin(seg, data_cfg["wt_labels"])
    return np.stack([et, tc, wt]).astype(np.float32)
