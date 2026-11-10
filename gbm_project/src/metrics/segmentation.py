"""Dice, IoU and 95th-percentile Hausdorff distance for binary 3D masks (numpy bool arrays)."""
import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice_iou(pred, gt):
    pred, gt = pred.astype(bool), gt.astype(bool)
    inter = (pred & gt).sum()
    p, g = pred.sum(), gt.sum()
    if p + g == 0:                       # both empty = perfect
        return 1.0, 1.0
    return 2.0 * inter / (p + g), inter / (p + g - inter)


def hd95(pred, gt, spacing=(1.0, 1.0, 1.0)):
    pred, gt = pred.astype(bool), gt.astype(bool)
    if not pred.any() and not gt.any():
        return 0.0
    if not pred.any() or not gt.any():
        return np.nan
    surf = lambda m: m & ~binary_erosion(m)
    d_to_gt = distance_transform_edt(~gt, sampling=spacing)
    d_to_pred = distance_transform_edt(~pred, sampling=spacing)
    d = np.hstack([d_to_gt[surf(pred)], d_to_pred[surf(gt)]])
    return float(np.percentile(d, 95))


REGIONS = ["ET", "TC", "WT"]


def region_metrics(pred_bin, gt_bin, spacing=(1.0, 1.0, 1.0), with_hd=False):
    """pred_bin / gt_bin: (3,D,H,W) arrays. Returns {'ET': {...}, ...}"""
    out = {}
    for i, name in enumerate(REGIONS):
        d, j = dice_iou(pred_bin[i], gt_bin[i])
        out[name] = {"dice": d, "iou": j}
        if with_hd:
            out[name]["hd95"] = hd95(pred_bin[i], gt_bin[i], spacing)
    return out
