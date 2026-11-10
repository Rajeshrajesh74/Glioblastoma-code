import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceBCELoss(nn.Module):
    """Soft Dice + BCE on the 3 multi-label channels (ET, TC, WT)."""

    def forward(self, logits, target):
        bce = F.binary_cross_entropy_with_logits(logits, target)
        p = torch.sigmoid(logits)
        dims = (2, 3, 4)
        inter = (p * target).sum(dims)
        denom = p.sum(dims) + target.sum(dims)
        dice = (2 * inter + 1.0) / (denom + 1.0)
        return bce + (1.0 - dice.mean())
