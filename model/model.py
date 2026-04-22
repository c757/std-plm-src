from typing import Iterator, Mapping
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple, Union
from utils.utils import norm_Adj, lap_eig, topological_sort
# from model.sandglassAttn import SAG, CrossModalityAlignment
from model.topology_gcn import TopologyGCNBridge
from model.RevIN import RevIN
import numpy as np
from model.position import PositionalEncoding
from torch.utils.checkpoint import checkpoint


class TimeEmbedding(nn.Module):

    def __init__(self, t_dim):
        super().__init__()

        # 周期函数时间编码（重点覆盖潮汐周期）
        # M2 主半日潮: 12.42h；同时加入日/周相关周期提升稳健性
        period_minutes = [
            12.42 * 60,  # M2
            12.00 * 60,  # S2
            23.93 * 60,  # K1
            25.82 * 60,  # O1
            24.00 * 60,  # solar-day
            7 * 24 * 60  # week
        ]
        self.register_buffer("period_minutes", torch.tensor(period_minutes, dtype=torch.float32))
        in_dim = len(period_minutes) * 2

        self.trig_proj = nn.Sequential(
            nn.Linear(in_dim, t_dim * 2),
            nn.GELU(),
            nn.Linear(t_dim * 2, t_dim * 2),
        )
        self.ln = nn.LayerNorm(t_dim * 2)

    def forward(self, TE):
        # TE (B,T,5)

        B, T, _ = TE.shape

        week = (TE[..., 2].to(torch.long) % 7).view(B * T, -1)
        hour = (TE[..., 3].to(torch.long) % 24).view(B * T, -1)
        minute = (TE[..., 4].to(torch.long) % 60).view(B * T, -1)

        # 以“周内分钟”构造连续时间，相比固定 slot 更易建模潮汐相位漂移
        t_minutes = (week * 24 * 60 + hour * 60 + minute).float()  # (B*T, 1)
        period = self.period_minutes.view(1, -1).to(t_minutes.device)  # (1, P)
        angle = 2 * math.pi * t_minutes / period  # (B*T, P)

        trig = torch.concat((torch.sin(angle), torch.cos(angle)), dim=-1)  # (B*T, 2P)
        te = self.ln(self.trig_proj(trig)).view(B, T, -1)

        return te


class NodeEmbedding(nn.Module):
    # 接收我们在 DataLoader 算好的 node_embeddings，抛弃 adj_mx
    def __init__(self, node_embeddings, node_emb_dim, k=16, dropout=0):
        super().__init__()
        self.k = k
        # 将传入的特征向量注册为不可训练的 buffer
        self.register_buffer('lap_eigvec', node_embeddings.float())
        self.fc = nn.Linear(in_features=k, out_features=node_emb_dim)

    def forward(self):
        node_emgedding = self.fc(self.lap_eigvec)
        return node_emgedding

    # 注意：原本这里还有一个 setadj(self, adj_mx) 方法，彻底删掉它！


class Time2Token(nn.Module):
    def __init__(self, sample_len, features, emb_dim, tim_dim, dropout):
        super().__init__()

        self.sample_len = sample_len

        in_features = sample_len * features * 2 + tim_dim
        hidden_size = (in_features + emb_dim) * 2 // 3
        self.fc_state = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, emb_dim),
        )

        input_dim = tim_dim + (sample_len - 1) * features * 2
        hidden_size = (input_dim + emb_dim) * 2 // 3
        self.fc_grad = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, emb_dim),
        )

        self.ln = nn.LayerNorm(emb_dim)


# 以下直接展示最重要的 MIMO 改造部分：

class Node2Token_Independent(nn.Module):
    """
    为单一物理要素专门定制的 Tokenizer，保证通道独立性（Channel Independence）
    """

    def __init__(self, sample_len, feature_dim, node_emb_dim, emb_dim, tim_dim, dropout, use_node_embedding):
        super().__init__()
        in_features = sample_len * feature_dim * 2  # 包含 mask 拼接
        self.use_node_embedding = use_node_embedding
        state_features = tim_dim
        if use_node_embedding:
            state_features += node_emb_dim

        self.fc1 = nn.Sequential(nn.Linear(in_features, emb_dim))

        self.state_fc = nn.Sequential(
            nn.Linear(state_features, node_emb_dim),
            nn.ReLU(),
            nn.Linear(node_emb_dim, emb_dim),
        )
        self.mask_token = nn.Linear(in_features=sample_len * feature_dim, out_features=emb_dim)
        self.ln = nn.LayerNorm(emb_dim)

    def forward(self, x, te, ne, mask):
        B, N, TF = x.shape
        mask = mask.contiguous().view(B, N, -1)
        x = torch.concat((x, mask), dim=-1)

        state = te[:, -1:, :].repeat(1, N, 1)
        if self.use_node_embedding:
            ne = torch.unsqueeze(ne, dim=0).repeat(B, 1, 1)
            state = torch.concat((state, ne), dim=-1)
        state = self.state_fc(state)

        x = self.fc1(x)
        x += self.mask_token(mask)
        out = self.ln(state + x)
        return out


class Node2Token_MultiScaleCNN(nn.Module):
    """
    Multi-scale CNN tokenizer.
    - Spatial path: (B,T,C,H,W) -> Conv2d branches -> (B,N,emb_dim)
    - Legacy path: (B,N,T*C) -> Conv1d branches -> (B,N,emb_dim)
    """

    def __init__(self, sample_len, feature_dim, node_emb_dim, emb_dim, tim_dim, dropout, use_node_embedding):
        super().__init__()
        self.sample_len = sample_len
        self.feature_dim = feature_dim
        self.emb_dim = emb_dim
        self.use_node_embedding = use_node_embedding

        in_ch = feature_dim * 2  # x + mask
        half = emb_dim // 2
        rest = emb_dim - half

        # Spatial multi-scale branches
        self.spatial_k3 = nn.Sequential(
            nn.Conv2d(in_ch, half, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.spatial_k7 = nn.Sequential(
            nn.Conv2d(in_ch, rest, kernel_size=7, padding=3),
            nn.GELU(),
        )

        # Legacy (flattened) temporal branches
        self.temporal_k3 = nn.Sequential(
            nn.Conv1d(in_ch, half, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.temporal_k5 = nn.Sequential(
            nn.Conv1d(in_ch, rest, kernel_size=5, padding=2),
            nn.GELU(),
        )

        state_features = tim_dim + (node_emb_dim if use_node_embedding else 0)
        self.state_fc = nn.Sequential(
            nn.Linear(state_features, node_emb_dim),
            nn.ReLU(),
            nn.Linear(node_emb_dim, emb_dim),
        )
        self.out_ln = nn.LayerNorm(emb_dim)

    def _state_token(self, te, ne, B, N):
        T = te.shape[1]
        # te: (B, T, tim_dim) -> (B, T, N, tim_dim)
        state = te.unsqueeze(2).repeat(1, 1, N, 1)
        if self.use_node_embedding:
            # ne: (N, node_emb_dim) -> (B, T, N, node_emb_dim)
            ne = ne.unsqueeze(0).unsqueeze(0).repeat(B, T, 1, 1)
            state = torch.concat((state, ne), dim=-1)
        return self.state_fc(state)  # (B, T, N, emb_dim)

    def _legacy_tokenize(self, x, mask):
        B, N, TF = x.shape
        x4 = x.contiguous().view(B, N, self.sample_len, self.feature_dim).permute(0, 1, 3, 2)  # (B,N,C,T)
        m4 = mask.contiguous().view(B, N, self.sample_len, self.feature_dim).permute(0, 1, 3, 2)
        z = torch.concat((x4, m4), dim=2).contiguous().view(B * N, self.feature_dim * 2, self.sample_len)

        y = torch.concat((self.temporal_k3(z), self.temporal_k5(z)), dim=1)  # (B*N, emb, T)
        y = y.permute(0, 2, 1).contiguous().view(B, N, self.sample_len, self.emb_dim)
        y = y.permute(0, 2, 1, 3).contiguous()  # (B, T, N, emb_dim)
        return y

    def _spatial_tokenize(self, x_spatial, mask_spatial):
        # x_spatial/mask_spatial: (B,T,C,H,W)
        B, T, C, H, W = x_spatial.shape
        z = torch.concat((x_spatial, mask_spatial), dim=2).contiguous().view(B * T, C * 2, H, W)

        y = torch.concat((self.spatial_k3(z), self.spatial_k7(z)), dim=1)  # (B*T,emb,H,W)
        y = y.view(B, T, self.emb_dim, H, W)
        y = y.permute(0, 1, 3, 4, 2).contiguous().view(B, T, H * W, self.emb_dim)
        return y  # (B, T, N, emb_dim)

    def forward(self, x, te, ne, mask, x_spatial=None, mask_spatial=None):
        B, N, _ = x.shape
        if x_spatial is not None and mask_spatial is not None:
            token = self._spatial_tokenize(x_spatial, mask_spatial)  # (B, T, N, D)
        else:
            token = self._legacy_tokenize(x, mask)  # (B, T, N, D)

        state = self._state_token(te, ne, B, N)  # (B, T, N, D)
        return self.out_ln(state + token)  # (B, T, N, D)


class DecodingLayer(nn.Module):
    def __init__(self, emb_dim, output_dim):
        super().__init__()
        hidden_size = (emb_dim + output_dim) * 2 // 3
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )

    def forward(self, llm_hidden):
        return self.fc(llm_hidden)


class DynamicTriModalFusion(nn.Module):
    """
    三模态动态融合：每个变量都用自身作为主干，并按余弦相似度动态吸收另外两种变量信息。
    输出保持变量独立（flow/wave/wind 各有各的融合结果），同时显式建模相互影响。
    """

    def __init__(self, emb_dim, dropout=0.0, temperature=0.5, mode="cosine"):
        super().__init__()
        if mode not in ("cosine", "qkv"):
            raise ValueError(f"Unsupported fusion mode: {mode}. Expected 'cosine' or 'qkv'.")

        self.mode = mode
        self.temperature = temperature
        self.scale = emb_dim ** -0.5
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(3)])
        self.ln = nn.ModuleList([nn.LayerNorm(emb_dim) for _ in range(3)])

        # QKV 融合专用投影（按“目标变量”分别建模）
        self.q_proj = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(3)])
        self.k_proj = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(3)])
        self.v_proj = nn.ModuleList([nn.Linear(emb_dim, emb_dim) for _ in range(3)])

    def _fuse_one(self, core, all_states, proj, ln):
        # core: (B, S, D), all_states: list[(B, S, D)]
        sims = [F.cosine_similarity(core, s, dim=-1).unsqueeze(-1) for s in all_states]  # 3 x (B, S, 1)
        sims = torch.concat(sims, dim=-1) / self.temperature  # (B, S, 3)
        weights = torch.softmax(sims, dim=-1)

        # 显存优化：废弃 torch.stack，直接逐个矩阵加权相加
        # 完美规避了在内存中生成 (B, S, 3, D) 这种极其浪费显存的中间变量
        fused = weights[..., 0:1] * all_states[0] + \
                weights[..., 1:2] * all_states[1] + \
                weights[..., 2:3] * all_states[2]

        out = ln(core + self.dropout(proj(fused)))
        return out, weights

    def _fuse_one_qkv(self, core, all_states, idx):
        # core: (B, S, D), all_states: list[(B, S, D)]
        stacked = torch.stack(all_states, dim=-2)  # (B, S, 3, D)

        q = self.q_proj[idx](core).unsqueeze(-2)  # (B, S, 1, D)
        k = self.k_proj[idx](stacked)  # (B, S, 3, D)
        v = self.v_proj[idx](stacked)  # (B, S, 3, D)

        scores = torch.sum(q * k, dim=-1) * self.scale
        weights = torch.softmax(scores, dim=-1)  # (B, S, 3)
        fused = torch.sum(weights.unsqueeze(-1) * v, dim=-2)  # (B, S, D)

        out = self.ln[idx](core + self.dropout(self.proj[idx](fused)))
        return out, weights

    def forward(self, s_flow, s_wave, s_wind):
        all_states = [s_flow, s_wave, s_wind]

        if self.mode == "qkv":
            flow_out, w_flow = self._fuse_one_qkv(s_flow, all_states, idx=0)
            wave_out, w_wave = self._fuse_one_qkv(s_wave, all_states, idx=1)
            wind_out, w_wind = self._fuse_one_qkv(s_wind, all_states, idx=2)
        else:
            flow_out, w_flow = self._fuse_one(s_flow, all_states, self.proj[0], self.ln[0])
            wave_out, w_wave = self._fuse_one(s_wave, all_states, self.proj[1], self.ln[1])
            wind_out, w_wind = self._fuse_one(s_wind, all_states, self.proj[2], self.ln[2])

        weight_dict = {"flow": w_flow, "wave": w_wave, "wind": w_wind}
        return (flow_out, wave_out, wind_out), weight_dict


class STALLM_MIMO(nn.Module):
    """
    支持物理分群（流、浪、風）的多輸入多輸出時空架構
    """

    def __init__(self, basemodel, sample_len, output_len,
                 input_dim, output_dim, node_emb_dim,
                 node_embeddings=None, use_node_embedding=True,
                 use_gcn=True,
                 dropout=0, trunc_k=16, t_dim=64, fusion_mode="cosine",
                 use_revin=False, revin_affine=True, edge_index=None,use_fp16=False):
        super().__init__()

        self.use_fp16 = use_fp16
        self.sample_len = sample_len
        self.output_len = output_len
        self.emb_dim = basemodel.emb_dim
        self.basemodel = basemodel
        # self.use_sandglassAttn = use_sandglassAttn
        self.use_gcn = use_gcn
        self.use_node_embedding = use_node_embedding
        self.fusion_mode = fusion_mode
        self.use_revin = use_revin

        tim_dim = t_dim * 2

        # 1. 物理維度定義 (根據 clear.py 的 0,1 | 2,3,4,5 | 6,7 分佈)
        self.dims = {'flow': 2, 'wave': 4, 'wind': 2}

        # 2. 為每一組物理量配置獨立的編碼器
        from model.model import Node2Token_MultiScaleCNN, TimeEmbedding, NodeEmbedding, DecodingLayer

        self.tokenizer_flow = Node2Token_MultiScaleCNN(sample_len, self.dims['flow'], node_emb_dim, self.emb_dim,
                                                       tim_dim, dropout, use_node_embedding)
        self.tokenizer_wave = Node2Token_MultiScaleCNN(sample_len, self.dims['wave'], node_emb_dim, self.emb_dim,
                                                       tim_dim, dropout, use_node_embedding)
        self.tokenizer_wind = Node2Token_MultiScaleCNN(sample_len, self.dims['wind'], node_emb_dim, self.emb_dim,
                                                       tim_dim, dropout, use_node_embedding)

        # 3. 拓扑感知空间传导模块（GCN）
        # if use_sandglassAttn:
        if use_gcn:
            self.gcn_flow = TopologyGCNBridge(self.emb_dim, edge_index=edge_index, dropout=dropout)
            self.gcn_wave = TopologyGCNBridge(self.emb_dim, edge_index=edge_index, dropout=dropout)
            self.gcn_wind = TopologyGCNBridge(self.emb_dim, edge_index=edge_index, dropout=dropout)

        # 三模态动态融合（无论是否启用 GCN，都在 token 空间融合）
        self.dynamic_fusion = DynamicTriModalFusion(self.emb_dim, dropout=dropout, mode=fusion_mode)

        # 4. 並行多頭解碼器
        self.head_flow = DecodingLayer(self.emb_dim, self.dims['flow'] * output_len)
        self.head_wave = DecodingLayer(self.emb_dim, self.dims['wave'] * output_len)
        self.head_wind = DecodingLayer(self.emb_dim, self.dims['wind'] * output_len)

        # 可逆实例归一化（按变量分组）
        if self.use_revin:
            self.revin_flow = RevIN(num_features=self.dims['flow'], affine=revin_affine)
            self.revin_wave = RevIN(num_features=self.dims['wave'], affine=revin_affine)
            self.revin_wind = RevIN(num_features=self.dims['wind'], affine=revin_affine)

        # 基礎組件
        self.timeembedding = TimeEmbedding(t_dim=t_dim)
        if use_node_embedding and node_embeddings is not None:
            self.node_embd_layer = NodeEmbedding(node_embeddings, node_emb_dim, trunc_k, dropout)
            self.llmSpatialProj = nn.Linear(node_emb_dim, self.emb_dim)

    def forward(self, x, timestamp, prompt_prefix, mask):
        # Commit-1 adapter: accept both legacy 3D input and new 5D grid input.
        x_spatial = None
        m_spatial = None
        if x.ndim == 5:
            # x/mask: (B, T, C, H, W) -> legacy (B, N, T*C)
            B, T, C, H, W = x.shape
            N = H * W
            x_spatial = x[:, :self.sample_len]
            if mask is not None and mask.ndim == 5:
                m_spatial = mask[:, :self.sample_len]
            x_flat = x.permute(0, 3, 4, 1, 2).contiguous().view(B, N, T * C)
            if mask is None:
                mask_flat = torch.ones_like(x_flat)
            elif mask.ndim == 5:
                mask_flat = mask.permute(0, 3, 4, 1, 2).contiguous().view(B, N, T * C)
            else:
                mask_flat = mask
        elif x.ndim == 3:
            B, N, TF = x.shape
            x_flat = x
            mask_flat = mask if mask is not None else torch.ones_like(x_flat)
        else:
            raise ValueError(f"Unsupported input ndim={x.ndim}, expected 3 or 5")

        x_reshaped = x_flat.view(B, N, self.sample_len, -1)
        mask_reshaped = mask_flat.view(B, N, self.sample_len, -1)

        # 物理索引精確切片 (对齐 8 维新数据)
        x_f_4d = x_reshaped[..., [0, 1]].contiguous()
        m_f = mask_reshaped[..., [0, 1]].contiguous().view(B, N, -1)

        # 波浪增加第 5 维 (包含 sin 和 cos)
        x_wa_4d = x_reshaped[..., [2, 3, 4, 5]].contiguous()
        m_wa = mask_reshaped[..., [2, 3, 4, 5]].contiguous().view(B, N, -1)

        # 海风被挤到了第 6, 7 维
        x_wi_4d = x_reshaped[..., [6, 7]].contiguous()
        m_wi = mask_reshaped[..., [6, 7]].contiguous().view(B, N, -1)

        # RevIN: 使用当前输入窗口统计量做归一化（per-batch, per-channel）
        if self.use_revin:
            x_f_4d = self.revin_flow(x_f_4d, 'norm')
            x_wa_4d = self.revin_wave(x_wa_4d, 'norm')
            x_wi_4d = self.revin_wind(x_wi_4d, 'norm')

        x_f = x_f_4d.view(B, N, -1)
        x_wa = x_wa_4d.view(B, N, -1)
        x_wi = x_wi_4d.view(B, N, -1)

        te = self.timeembedding(timestamp[:, :self.sample_len, :])
        ne = self.node_embd_layer() if self.use_node_embedding else None
        # 新增：预先计算好高维空间提示，准备广播给所有时间步
        if self.use_node_embedding and ne is not None:
            # ne: (N, node_emb_dim) -> (N, emb_dim) -> (1, 1, N, emb_dim)
            spatialHint = self.llmSpatialProj(ne).unsqueeze(0).unsqueeze(1)
        else:
            spatialHint = 0

        # 獨立編碼
        if x_spatial is not None and m_spatial is not None:
            tokens_f = self.tokenizer_flow(x_f, te, ne, m_f, x_spatial=x_spatial[:, :, [0, 1], :, :],
                                           mask_spatial=m_spatial[:, :, [0, 1], :, :])
            tokens_wa = self.tokenizer_wave(x_wa, te, ne, m_wa, x_spatial=x_spatial[:, :, [2, 3, 4, 5], :, :],
                                            mask_spatial=m_spatial[:, :, [2, 3, 4, 5], :, :])
            tokens_wi = self.tokenizer_wind(x_wi, te, ne, m_wi, x_spatial=x_spatial[:, :, [6, 7], :, :],
                                            mask_spatial=m_spatial[:, :, [6, 7], :, :])
        else:
            tokens_f = self.tokenizer_flow(x_f, te, ne, m_f)
            tokens_wa = self.tokenizer_wave(x_wa, te, ne, m_wa)
            tokens_wi = self.tokenizer_wind(x_wi, te, ne, m_wi)

        # 动态融合 + 独立后半程
        if self.use_gcn:
            s_f = self.gcn_flow(tokens_f)
            s_wa = self.gcn_wave(tokens_wa)
            s_wi = self.gcn_wind(tokens_wi)

            B_f, T_f, N_f, D_f = s_f.shape

            # 优化：使用 Checkpoint 彻底压制融合阶段的显存堆积
            def wrapper_fusion(f, wa, wi):
                return self.dynamic_fusion(f, wa, wi)

            alignedFList, alignedWaList, alignedWiList = [], [], []
            weight_dict = {}
            for t in range(T_f):
                fT, waT, wiT = s_f[:, t, :, :], s_wa[:, t, :, :], s_wi[:, t, :, :]

                # 使用 checkpoint 包装：计算完立刻释放内存，反向传播再重算
                (aFT, aWaT, aWiT), wDict = checkpoint(wrapper_fusion, fT, waT, wiT, use_reentrant=False)

                alignedFList.append(aFT)
                alignedWaList.append(aWaT)
                alignedWiList.append(aWiT)
                if t == T_f - 1: weight_dict = wDict

            aligned_f = torch.stack(alignedFList, dim=1) + spatialHint
            aligned_wa = torch.stack(alignedWaList, dim=1) + spatialHint
            aligned_wi = torch.stack(alignedWiList, dim=1) + spatialHint

            # aligned_f/wa/wi: (B, T, N, D)
            # 让 LLM 沿时间方向处理：reshape 为 (B*N, T, D)
            B_cur, T_cur, N_cur, D_cur = aligned_f.shape

            # 统一的大模型推演逻辑
            def _run_llm(feat):
                b_f, t_f, n_f, d_f = feat.shape
                # 转换形状为 (B*N, T, D)
                inp = feat.permute(0, 2, 1, 3).contiguous().view(b_f * n_f, t_f, d_f)

                # 🚀 降维打击：将 chunk_size 压到 1024，显存压力再降 4 倍
                chunk_size = 1024
                out_list = []

                for i in range(0, inp.shape[0], chunk_size):
                    chunk = inp[i: i + chunk_size]

                    # 核心：在 chunk 级别开启 checkpoint
                    # 这样 PyTorch 就不会在显存里堆积这 15x3=45 个 chunk 的 Transformer 激活值了
                    def chunk_forward(c):
                        # 必须在 checkpoint 内部重新包裹 autocast，否则 FP16 会失效
                        with torch.amp.autocast('cuda', enabled=self.use_fp16):
                            return self.basemodel(c)

                    # 执行 checkpoint
                    chunk_out = checkpoint(chunk_forward, chunk, use_reentrant=False)
                    out_list.append(chunk_out)

                out = torch.cat(out_list, dim=0)
                # 还原形状为 (B, T, N, D)
                return out.view(b_f, n_f, t_f, d_f).permute(0, 2, 1, 3).contiguous()

            hidden_f = _run_llm(aligned_f)
            hidden_wa = _run_llm(aligned_wa)
            hidden_wi = _run_llm(aligned_wi)

            decoded_f = hidden_f + s_f
            decoded_wa = hidden_wa + s_wa
            decoded_wi = hidden_wi + s_wi
        else:
            B_f, T_f, N_f, D_f = tokens_f.shape

            # 显存优化：同上
            alignedFList = []
            alignedWaList = []
            alignedWiList = []
            weight_dict = {}
            for t in range(T_f):
                fT = tokens_f[:, t, :, :]
                waT = tokens_wa[:, t, :, :]
                wiT = tokens_wi[:, t, :, :]
                (aFT, aWaT, aWiT), wDict = self.dynamic_fusion(fT, waT, wiT)
                alignedFList.append(aFT)
                alignedWaList.append(aWaT)
                alignedWiList.append(aWiT)
                if t == T_f - 1:
                    weight_dict = wDict

            aligned_f = torch.stack(alignedFList, dim=1) + spatialHint
            aligned_wa = torch.stack(alignedWaList, dim=1) + spatialHint
            aligned_wi = torch.stack(alignedWiList, dim=1) + spatialHint

            B_cur, T_cur, N_cur, D_cur = aligned_f.shape

            def _run_llm(feat):
                inp = feat.permute(0, 2, 1, 3).contiguous().view(B_cur * N_cur, T_cur, D_cur)
                out = self.basemodel(inp)
                return out.view(B_cur, N_cur, T_cur, D_cur).permute(0, 2, 1, 3).contiguous()

            hidden_f = _run_llm(aligned_f)
            hidden_wa = _run_llm(aligned_wa)
            hidden_wi = _run_llm(aligned_wi)

            decoded_f = hidden_f + tokens_f
            decoded_wa = hidden_wa + tokens_wa
            decoded_wi = hidden_wi + tokens_wi

        # decoded: (B, T, N, D) -> 取最后一个时间步作为预测起点
        # 这与 LLM 自回归输出习惯一致：用序列末尾 token 做预测
        def _decode(decoded, head, out_len, feat_dim):
            # 取最后一个时间步: (B, T, N, D) -> (B, N, D)
            last = decoded[:, -1, :, :]
            return head(last).view(B, N, out_len, feat_dim)

        pred_flow = _decode(decoded_f, self.head_flow, self.output_len, self.dims['flow'])
        pred_wave = _decode(decoded_wa, self.head_wave, self.output_len, self.dims['wave'])
        pred_wind = _decode(decoded_wi, self.head_wind, self.output_len, self.dims['wind'])

        # RevIN 反归一化：将预测恢复到输入窗口对应的原始尺度
        if self.use_revin:
            pred_flow = self.revin_flow(pred_flow, 'denorm')
            pred_wave = self.revin_wave(pred_wave, 'denorm')
            pred_wind = self.revin_wind(pred_wind, 'denorm')

        return {
            "flow": pred_flow.contiguous().view(B, N, -1),
            "wave": pred_wave.contiguous().view(B, N, -1),
            "wind": pred_wind.contiguous().view(B, N, -1)
        }, [], {"fusion_weights": weight_dict}

    def params_num(self):
        total = sum(p.numel() for p in self.parameters()) + sum(p.numel() for p in self.buffers())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def grad_state_dict(self):
        return {n: p.detach() for n, p in self.named_parameters() if p.requires_grad}

    def save(self, path: str):
        torch.save(self.grad_state_dict(), path)

    def load(self, path: str):
        self.load_state_dict(torch.load(path), strict=False)
