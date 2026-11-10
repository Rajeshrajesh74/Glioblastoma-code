"""Time-aware temporal Transformer: elapsed time between scans enters the attention itself."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEncoding(nn.Module):
    """Continuous sinusoidal encoding of 'days since first scan' (irregular spacing OK)."""

    def __init__(self, d):
        super().__init__()
        assert d % 2 == 0
        self.register_buffer("freqs", torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d)))
        self.mlp = nn.Linear(d, d)

    def forward(self, t_days):                                            # (B,T)
        a = (t_days / 30.0).unsqueeze(-1) * self.freqs                     # months
        return self.mlp(torch.cat([a.sin(), a.cos()], -1))


class TimeAwareSelfAttention(nn.Module):
    """softmax(QK^T/sqrt(d) - decay_h * |t_i - t_j|):  learnable per-head time decay (per month)."""

    def __init__(self, d, heads, dropout=0.1):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.drop = nn.Dropout(dropout)
        self.decay_raw = nn.Parameter(torch.full((heads,), -2.0))

    def forward(self, x, t, valid):
        B, T, d = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.h, self.dh).permute(2, 0, 3, 1, 4).unbind(0)   # B,h,T,dh
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        dt = (t[:, :, None] - t[:, None, :]).abs() / 30.0                                      # B,T,T months
        att = att - F.softplus(self.decay_raw).view(1, -1, 1, 1) * dt[:, None]
        att = att.masked_fill(~valid[:, None, None, :], float("-inf"))
        att = self.drop(att.softmax(-1))
        return self.proj((att @ v).transpose(1, 2).reshape(B, T, d))


class TemporalBlock(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = TimeAwareSelfAttention(d, heads, dropout)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout), nn.Linear(4 * d, d), nn.Dropout(dropout))

    def forward(self, x, t, valid):
        x = x + self.attn(self.n1(x), t, valid)
        return x + self.ff(self.n2(x))


class TimeAwareTransformer(nn.Module):
    def __init__(self, d, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.time_enc = TimeEncoding(d)
        self.blocks = nn.ModuleList([TemporalBlock(d, heads, dropout) for _ in range(layers)])
        self.norm = nn.LayerNorm(d)

    def forward(self, x, t, valid):                                       # x:(B,T,d) t:(B,T) valid:(B,T) bool
        x = x + self.time_enc(t)
        for blk in self.blocks:
            x = blk(x, t, valid)
        return self.norm(x)
