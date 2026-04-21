import torch
import torch.nn as nn


class TopologyGCNLayer(nn.Module):
    """Sparse graph convolution on node tokens: X' = W * mean_neighbors(X)."""

    def __init__(self, emb_dim, edge_index, dropout=0.0):
        super().__init__()
        self.emb_dim = emb_dim
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(emb_dim, emb_dim)
        self.ln = nn.LayerNorm(emb_dim)

        if edge_index is None:
            raise ValueError("edge_index must be provided for TopologyGCNLayer")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError(f"edge_index shape must be (2,E), got {tuple(edge_index.shape)}")

        self.register_buffer("edge_index", edge_index.long())

    def forward(self, x):
        # x: (B, N, D)
        bsz, n_nodes, dim = x.shape
        src = self.edge_index[0]
        dst = self.edge_index[1]

        if src.numel() == 0:
            return x

        agg = torch.zeros_like(x)
        deg = torch.zeros((n_nodes,), device=x.device, dtype=x.dtype)

        # index_add over node dimension (dim=1)
        agg.index_add_(1, dst, x[:, src, :])
        deg.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype, device=x.device))
        deg = deg.clamp_min(1.0).view(1, n_nodes, 1)

        neigh = agg / deg
        out = self.ln(x + self.dropout(self.proj(neigh)))
        return out


class TopologyGCNBridge(nn.Module):
    """Two-layer residual GCN bridge, input/output both (B,N,D)."""

    def __init__(self, emb_dim, edge_index, dropout=0.0):
        super().__init__()
        self.gcn1 = TopologyGCNLayer(emb_dim, edge_index, dropout=dropout)
        self.gcn2 = TopologyGCNLayer(emb_dim, edge_index, dropout=dropout)

    def forward(self, x):
        # x: (B, T, N, D)
        B, T, N, D = x.shape
        x = x.view(B * T, N, D)
        x = self.gcn1(x)
        x = self.gcn2(x)
        return x.view(B, T, N, D)
