import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.position import PositionalEncoding

class SAG(nn.Module):
    """
    单要素的沙漏注意力编码器（负责将 63200 个空间节点压缩为 128 个超节点）
    """
    def __init__(self, sag_dim, sag_tokens, emb_dim, sample_len, features, dropout):
        super().__init__()
        self.sag_tokens = sag_tokens
        self.num_heads = 4
        self.sag_dim = sag_dim

        self.hyper_nodes = nn.Parameter(torch.randn(1,sag_tokens,sag_dim))
        self.pe = PositionalEncoding(num_hiddens=sag_dim,dropout=dropout,max_len=65000)

        self.emc_mha = nn.MultiheadAttention(embed_dim=sag_dim,num_heads=self.num_heads,batch_first=True, dropout=dropout)
        self.dec_mha = nn.MultiheadAttention(embed_dim=sag_dim,num_heads=self.num_heads,batch_first=True, dropout=dropout,vdim=emb_dim)

        self.enc_fc = nn.Linear(in_features=sag_dim,out_features=emb_dim)
        self.dec_fc = nn.Linear(in_features=sag_dim,out_features=emb_dim)
        self.x_fc = nn.Linear(in_features=emb_dim,out_features=sag_dim)

        self.en_ln = nn.LayerNorm(emb_dim)
        self.de_ln = nn.LayerNorm(emb_dim)

    def encode(self,x):
        # x: (B, N, D)
        B,N,H = x.shape
        kv = self.x_fc(x)
        q = self.pe(self.hyper_nodes)

        # 压缩到 Hyper-nodes
        out,attn_weights = self.emc_mha(query=q.clone().repeat(B,1,1),key=self.pe(kv),value=kv.clone()) 
        out = self.enc_fc(out)
        out = self.en_ln(out)
        return out, attn_weights

    def decode(self,hidden_state,x):
        # 将大模型处理后的超节点映射回 N 个真实物理节点
        B,_,_ = hidden_state.shape
        q = self.pe(self.x_fc(x))
        k = self.pe(self.hyper_nodes)
        v = hidden_state

        out, _ = self.dec_mha(query=q, key=k.clone().repeat(B,1,1), value=v.clone())
        out = self.dec_fc(out)
        out = self.de_ln(out)
        return out

class CrossModalityAlignment(nn.Module):
    """
    【全新模块】：跨模态物理对齐模块 (TimeCMA 核心思想)
    在 128 个超节点的高维空间中，计算外源环境（如风、浪）对核心变量（如流）的注意力驱动。
    """
    def __init__(self, emb_dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.cross_mha = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=num_heads, batch_first=True, dropout=dropout)
        self.ln1 = nn.LayerNorm(emb_dim)
        self.ffn = nn.Sequential(
            nn.Linear(emb_dim, emb_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(emb_dim * 2, emb_dim)
        )
        self.ln2 = nn.LayerNorm(emb_dim)

    def forward(self, core_state, context_state):
        """
        :param core_state: 核心物理量（如洋流），作为 Query
        :param context_state: 环境变量（如海风），作为 Key 和 Value
        """
        # 计算风对流的影响
        attn_out, _ = self.cross_mha(query=core_state, key=context_state, value=context_state)
        # 残差连接：流自身规律 + 风的驱动扰动
        out = self.ln1(core_state + attn_out) 
        
        # FFN 前馈网络增强非线性
        ffn_out = self.ffn(out)
        out = self.ln2(out + ffn_out)
        
        return out