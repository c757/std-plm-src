import torch
import torch.nn as nn


class TopologyGCNLayer(nn.Module):
    """Sparse graph convolution on node tokens using highly optimized Sparse MatMul."""

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

        # 显存优化核心：预先计算并缓存“归一化的稀疏邻接矩阵”
        src = edge_index[0]
        dst = edge_index[1]

        # 动态获取节点总数
        n_nodes = max(src.max(), dst.max()).item() + 1

        # 1. 构建全 1 的基础稀疏邻接矩阵 (COO格式)
        val = torch.ones_like(src, dtype=torch.float32)
        sparseAdj = torch.sparse_coo_tensor(edge_index, val, (n_nodes, n_nodes))

        # 2. 计算每个节点的入度 (Degree)
        deg = torch.sparse.sum(sparseAdj, dim=1).to_dense()
        deg = deg.clamp_min(1.0)  # 防止除零

        # 3. 归一化：A_norm[dst, src] = 1 / deg[dst]
        # 在 COO 格式中，直接用值除以目标节点的度数即可
        valNorm = val / deg[dst]

        # 4. 生成最终的归一化稀疏矩阵并 coalesce (合并重复索引加速计算)
        sparseNormAdj = torch.sparse_coo_tensor(edge_index, valNorm, (n_nodes, n_nodes)).coalesce()

        # 将其注册为模型的不可训练 Buffer，它会自动跟随模型 save/load 并迁移到正确的设备 (GPU/CPU)
        self.register_buffer("sparseNormAdj", sparseNormAdj)

    def forward(self, x):
        # x: (B, N, D)
        B, N, D = x.shape

        # 记录进入 GCN 之前的类型 (通常是 float16)
        originalDtype = x.dtype

        # 显存与精度优化：强制关闭这一块的自动混合精度拦截
        with torch.amp.autocast('cuda', enabled=False):
            # 此时系统绝对服从 float32 指令
            xFloat32 = x.to(torch.float32)
            adjFloat32 = self.sparseNormAdj.to(torch.float32)

            xReshaped = xFloat32.transpose(0, 1).reshape(N, B * D)

            # 极速稀疏矩阵乘法，绝对不会再触发 "Half" 报错
            neighReshaped = torch.sparse.mm(adjFloat32, xReshaped)

        # 离开安全区后，立刻转回原来的类型，交还给后续网络继续加速
        neigh = neighReshaped.view(N, B, D).transpose(0, 1).to(originalDtype)

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

        # 【内存优化核心】：原来是把 9 个时间步压平一起算，导致单次激活值极其巨大。
        # 现在改成按时间步循环处理，用时间换空间
        out_list = []
        for t in range(T):
            xt = x[:, t, :, :]  # 取出当前时间步: (B, N, D)
            xt = self.gcn1(xt)
            xt = self.gcn2(xt)
            out_list.append(xt)

        return torch.stack(out_list, dim=1)  # 重新拼回 (B, T, N, D)
