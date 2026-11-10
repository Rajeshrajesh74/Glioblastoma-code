"""Longitudinal model: region graphs -> GAT -> time-aware Transformer -> gated fusion -> 2 heads."""
import torch
import torch.nn as nn

from src.models.fusion import CrossAttentionGatedFusion
from src.models.gat import GraphEncoder
from src.models.temporal_transformer import TimeAwareTransformer


def mlp_head(d, dropout):
    return nn.Sequential(nn.Linear(d, d // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d // 2, 1))


class LongitudinalGBMModel(nn.Module):
    def __init__(self, node_in, n_clin, d_model=128, gat_heads=4, gat_layers=2,
                 temporal_heads=4, temporal_layers=2, dropout=0.1):
        super().__init__()
        self.node_proj = nn.Sequential(nn.Linear(node_in, d_model), nn.LayerNorm(d_model), nn.GELU())
        self.graph = GraphEncoder(d_model, gat_heads, gat_layers, dropout)
        self.temporal = TimeAwareTransformer(d_model, temporal_heads, temporal_layers, dropout)
        self.fusion = CrossAttentionGatedFusion(d_model, n_clin, temporal_heads, dropout)
        self.prog_head = mlp_head(d_model, dropout)     # progression classification (logit)
        self.surv_head = mlp_head(d_model, dropout)     # Cox log-risk

    def forward(self, nodes, times, valid, clin, clin_mask):
        """nodes:(B,T,N,F) times:(B,T) days  valid:(B,T) bool  clin:(B,K)  clin_mask:(B,K)"""
        B, T, N, Fdim = nodes.shape
        x = self.node_proj(nodes.reshape(B * T, N, Fdim))
        v = self.graph(x).view(B, T, -1)                                   # one vector per examination
        h = self.temporal(v, times, valid)                                 # (B,T,d)
        last = valid.long().sum(1) - 1                                     # most recent valid visit
        rep = h[torch.arange(B, device=h.device), last]
        z = self.fusion(rep, clin, clin_mask)
        return {"prog_logit": self.prog_head(z).squeeze(-1), "risk": self.surv_head(z).squeeze(-1)}
