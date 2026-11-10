"""
3D Hybrid CNN - Swin Transformer segmentation network.

  CNN encoder (local spatial detail)  ->  3D shifted-window Swin blocks (global context)
  -> CNN decoder with skip connections -> 3 output channels: ET, TC, WT (sigmoid, multi-label)

Input size: every spatial dim must be divisible by 64 (e.g. 128x128x128).
"""
import torch
import torch.nn as nn


# ------------------------------------------------------------------ helpers
def window_partition(x, ws):
    """x: (B,D,H,W,C) -> (B*nW, N, C)"""
    B, D, H, W, C = x.shape
    x = x.view(B, D // ws[0], ws[0], H // ws[1], ws[1], W // ws[2], ws[2], C)
    return x.permute(0, 1, 3, 5, 2, 4, 6, 7).reshape(-1, ws[0] * ws[1] * ws[2], C)


def window_reverse(w, ws, B, D, H, W):
    C = w.shape[-1]
    x = w.view(B, D // ws[0], H // ws[1], W // ws[2], ws[0], ws[1], ws[2], C)
    return x.permute(0, 1, 4, 2, 5, 3, 6, 7).reshape(B, D, H, W, C)


class ConvBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(cin, cout, 3, stride, 1, bias=False), nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU(0.1, True),
            nn.Conv3d(cout, cout, 3, 1, 1, bias=False), nn.InstanceNorm3d(cout, affine=True), nn.LeakyReLU(0.1, True),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------------ Swin 3D
class WindowAttention3D(nn.Module):
    def __init__(self, dim, heads, ws):
        super().__init__()
        self.heads, self.ws = heads, ws
        self.scale = (dim // heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

        # relative position bias
        self.rpb_table = nn.Parameter(torch.zeros((2 * ws[0] - 1) * (2 * ws[1] - 1) * (2 * ws[2] - 1), heads))
        nn.init.trunc_normal_(self.rpb_table, std=0.02)
        coords = torch.stack(torch.meshgrid(*[torch.arange(w) for w in ws], indexing="ij")).flatten(1)  # 3,N
        rel = (coords[:, :, None] - coords[:, None, :]).permute(1, 2, 0).contiguous()                  # N,N,3
        rel[..., 0] += ws[0] - 1
        rel[..., 1] += ws[1] - 1
        rel[..., 2] += ws[2] - 1
        rel[..., 0] *= (2 * ws[1] - 1) * (2 * ws[2] - 1)
        rel[..., 1] *= (2 * ws[2] - 1)
        self.register_buffer("rp_index", rel.sum(-1), persistent=False)

    def forward(self, x, mask=None):
        Bw, N, C = x.shape
        qkv = self.qkv(x).reshape(Bw, N, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q * self.scale) @ k.transpose(-2, -1)                                   # Bw,h,N,N
        bias = self.rpb_table[self.rp_index.view(-1)].view(N, N, -1).permute(2, 0, 1)   # h,N,N
        attn = attn + bias.unsqueeze(0)
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(Bw // nW, nW, self.heads, N, N) + mask[None, :, None]
            attn = attn.view(Bw, self.heads, N, N)
        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(Bw, N, C)
        return self.proj(out)


class SwinBlock3D(nn.Module):
    def __init__(self, dim, heads, ws=(4, 4, 4), shift=False, mlp_ratio=2.0, drop=0.0):
        super().__init__()
        self.ws = ws
        self.shift = tuple(w // 2 for w in ws) if shift else (0, 0, 0)
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, heads, ws)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Dropout(drop),
                                 nn.Linear(int(dim * mlp_ratio), dim), nn.Dropout(drop))

    def _mask(self, D, H, W, device):
        if not any(self.shift):
            return None
        img = torch.zeros(1, D, H, W, 1, device=device)
        slices = [(slice(0, -w), slice(-w, -s), slice(-s, None)) for w, s in zip(self.ws, self.shift)]
        cnt = 0
        for d in slices[0]:
            for h in slices[1]:
                for w in slices[2]:
                    img[:, d, h, w, :] = cnt
                    cnt += 1
        mw = window_partition(img, self.ws).squeeze(-1)           # nW,N
        m = mw[:, None, :] - mw[:, :, None]
        return m.masked_fill(m != 0, -100.0)

    def forward(self, x):                                          # x: (B,D,H,W,C)
        B, D, H, W, C = x.shape
        shortcut = x
        x = self.norm1(x)
        if any(self.shift):
            x = torch.roll(x, shifts=tuple(-s for s in self.shift), dims=(1, 2, 3))
        xw = self.attn(window_partition(x, self.ws), self._mask(D, H, W, x.device))
        x = window_reverse(xw, self.ws, B, D, H, W)
        if any(self.shift):
            x = torch.roll(x, shifts=self.shift, dims=(1, 2, 3))
        x = shortcut + x
        return x + self.mlp(self.norm2(x))


class SwinStage(nn.Module):
    """Operates on (B,C,D,H,W) feature maps; alternates regular / shifted windows."""

    def __init__(self, dim, depth, heads, ws=4, drop=0.0):
        super().__init__()
        ws = (ws, ws, ws)
        self.blocks = nn.ModuleList([SwinBlock3D(dim, heads, ws, shift=(i % 2 == 1), drop=drop) for i in range(depth)])

    def forward(self, x):
        x = x.permute(0, 2, 3, 4, 1)
        for blk in self.blocks:
            x = blk(x)
        return x.permute(0, 4, 1, 2, 3).contiguous()


# ------------------------------------------------------------------ full network
class HybridCNNSwinSeg(nn.Module):
    def __init__(self, in_ch=4, out_ch=3, base=16, swin_depth=2, window=4, drop=0.0):
        super().__init__()
        c1, c2, c3, c4, c5 = base, base * 2, base * 4, base * 8, base * 16
        self.bottleneck_ch = c5
        # CNN encoder
        self.enc1 = ConvBlock(in_ch, c1)                 # 1/1
        self.enc2 = ConvBlock(c1, c2, stride=2)          # 1/2
        self.enc3 = ConvBlock(c2, c3, stride=2)          # 1/4
        self.enc4 = ConvBlock(c3, c4, stride=2)          # 1/8
        # Swin transformer stages (global context)
        self.swin4 = SwinStage(c4, swin_depth, heads=4, ws=window, drop=drop)
        self.down5 = nn.Sequential(nn.Conv3d(c4, c5, 2, stride=2), nn.InstanceNorm3d(c5, affine=True))  # 1/16
        self.swin5 = SwinStage(c5, swin_depth, heads=8, ws=window, drop=drop)
        # CNN decoder
        self.up4 = nn.ConvTranspose3d(c5, c4, 2, 2); self.dec4 = ConvBlock(c4 * 2, c4)
        self.up3 = nn.ConvTranspose3d(c4, c3, 2, 2); self.dec3 = ConvBlock(c3 * 2, c3)
        self.up2 = nn.ConvTranspose3d(c3, c2, 2, 2); self.dec2 = ConvBlock(c2 * 2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, 2, 2); self.dec1 = ConvBlock(c1 * 2, c1)
        self.head = nn.Conv3d(c1, out_ch, 1)

    def forward(self, x, return_feats=False):
        assert all(s % 64 == 0 for s in x.shape[2:]), "spatial dims must be divisible by 64"
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.swin4(self.enc4(e3))
        b = self.swin5(self.down5(e4))
        d4 = self.dec4(torch.cat([self.up4(b), e4], 1))
        d3 = self.dec3(torch.cat([self.up3(d4), e3], 1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], 1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], 1))
        logits = self.head(d1)
        return (logits, b) if return_feats else logits
