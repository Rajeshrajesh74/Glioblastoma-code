"""Dense Graph Attention Network (pure PyTorch, no torch_geometric needed - graphs have only 4 nodes)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.1):
        super().__init__()
        assert dim % heads == 0
        self.h, self.f = heads, dim // heads
        self.W = nn.Linear(dim, dim, bias=False)
        self.a_src = nn.Parameter(torch.empty(heads, self.f))
        self.a_dst = nn.Parameter(torch.empty(heads, self.f))
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, adj):
        """x: (B,N,dim); adj: (B,N,N) 1 where an edge exists."""
        B, N, _ = x.shape
        h = self.W(x).view(B, N, self.h, self.f)
        e_i = (h * self.a_src).sum(-1)                                   # B,N,h
        e_j = (h * self.a_dst).sum(-1)
        e = F.leaky_relu(e_i.unsqueeze(2) + e_j.unsqueeze(1), 0.2)       # B,N,N,h
        e = e.masked_fill(adj.unsqueeze(-1) == 0, float("-inf"))
        alpha = self.drop(e.softmax(dim=2))
        out = torch.einsum("bijh,bjhf->bihf", alpha, h).reshape(B, N, -1)
        return self.norm(x + F.elu(out))                                  # residual


class GraphEncoder(nn.Module):
    """Stack of GAT layers + attention read-out -> one vector per MRI examination."""

    def __init__(self, dim, heads=4, layers=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([GATLayer(dim, heads, dropout) for _ in range(layers)])
        self.score = nn.Linear(dim, 1)

    def forward(self, x):                                                 # (B,N,dim)
        B, N, _ = x.shape
        adj = torch.ones(B, N, N, device=x.device)                        # fully connected + self loops
        for layer in self.layers:
            x = layer(x, adj)
        w = self.score(x).softmax(dim=1)
        return (w * x).sum(1)                                             # (B,dim)
