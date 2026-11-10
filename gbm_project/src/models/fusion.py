"""Cross-attention gated multimodal fusion: imaging representation <-> clinical / molecular tokens."""
import torch
import torch.nn as nn


class CrossAttentionGatedFusion(nn.Module):
    def __init__(self, d, n_clin, heads=4, dropout=0.1):
        super().__init__()
        # each clinical scalar becomes a token: value * w_k + b_k   (missing values are masked out)
        self.w = nn.Parameter(torch.randn(n_clin, d) * 0.02)
        self.b = nn.Parameter(torch.zeros(n_clin, d))
        self.null = nn.Parameter(torch.zeros(1, 1, d))                    # always-available token (no NaNs if all missing)
        self.attn = nn.MultiheadAttention(d, heads, dropout=dropout, batch_first=True)
        self.out = nn.Sequential(nn.Linear(d, d), nn.GELU())
        self.gate = nn.Sequential(nn.Linear(2 * d, d), nn.Sigmoid())
        self.norm = nn.LayerNorm(d)

    def forward(self, img, clin, clin_mask):
        """img:(B,d)  clin:(B,K) values (0 where missing)  clin_mask:(B,K) 1 = available"""
        B = img.shape[0]
        tok = clin.unsqueeze(-1) * self.w + self.b                         # B,K,d
        tok = torch.cat([self.null.expand(B, -1, -1), tok], 1)
        avail = torch.cat([torch.ones(B, 1, dtype=torch.bool, device=img.device), clin_mask.bool()], 1)
        ctx, _ = self.attn(img.unsqueeze(1), tok, tok, key_padding_mask=~avail)
        ctx = self.out(ctx.squeeze(1))
        g = self.gate(torch.cat([img, ctx], -1))                           # how much clinical info to let in
        return self.norm(img + g * ctx)
