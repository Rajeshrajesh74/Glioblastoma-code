"""Turn a segmentation + bottleneck features into 4 graph nodes: ET, necrotic core, edema, whole tumour."""
import torch
import torch.nn.functional as F

NODE_NAMES = ["ET", "NEC", "ED", "WT"]


@torch.no_grad()
def region_nodes(logits, bott, voxel_vol_mm3):
    """
    logits: (B,3,D,H,W) -> [ET,TC,WT];  bott: (B,C,d,h,w);  voxel_vol_mm3: (B,)
    returns (B,4,C+4): pooled deep features, log volume (ml), centroid xyz in [0,1]
    """
    p = torch.sigmoid(logits.float())
    et, tc, wt = p[:, 0:1], p[:, 1:2], p[:, 2:3]
    regions = torch.cat([et, (tc - et).clamp(min=0), (wt - tc).clamp(min=0), wt], 1)      # B,4,D,H,W

    m_small = F.interpolate(regions, size=bott.shape[2:], mode="trilinear", align_corners=False)
    num = torch.einsum("bcdhw,brdhw->brc", bott.float(), m_small)
    den = m_small.sum((2, 3, 4)).unsqueeze(-1) + 1e-6
    feat = num / den                                                                       # B,4,C

    vol_ml = regions.sum((2, 3, 4)) * voxel_vol_mm3.view(-1, 1) / 1000.0
    logvol = torch.log1p(vol_ml).unsqueeze(-1)                                             # B,4,1

    tot = regions.sum((2, 3, 4)) + 1e-6
    cents = []
    for ax, n in enumerate(regions.shape[2:]):
        coord = torch.linspace(0, 1, n, device=regions.device)
        dims = tuple(a + 2 for a in range(3) if a != ax)
        cents.append((regions.sum(dims) * coord).sum(-1) / tot)                            # B,4
    cent = torch.stack(cents, -1)                                                          # B,4,3
    return torch.cat([feat, logvol, cent], -1)
