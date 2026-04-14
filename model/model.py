from typing import Iterator, Mapping
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple, Union
from utils.utils import norm_Adj, lap_eig, topological_sort
from model.sandglassAttn import SAG, CrossModalityAlignment
import numpy as np
from model.position import PositionalEncoding
class TimeEmbedding(nn.Module):

    def __init__(self,t_dim):
        super().__init__()

        #self.hour_embedding = nn.Embedding(num_embeddings=24,embedding_dim=t_dim)
        self.day_embedding = nn.Embedding(num_embeddings=288,embedding_dim=t_dim)
        self.week_embedding = nn.Embedding(num_embeddings=7,embedding_dim=t_dim)

    def forward(self,TE):

        # TE (B,T,5)

        B,T,_ = TE.shape

        week = (TE[...,2].to(torch.long) % 7).view(B*T,-1)
        hour = (TE[...,3].to(torch.long) % 24).view(B*T,-1)
        minute = (TE[...,4].to(torch.long) % 60).view(B*T,-1)

        DE = self.day_embedding((hour*60+minute)//5)
        #HE = self.hour_embedding(hour)
        WE = self.week_embedding(week)

        te = torch.concat((DE,WE),dim=-1).view(B,T,-1)

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
    def __init__(self,sample_len, features, emb_dim, tim_dim, dropout):
        super().__init__()
        
        self.sample_len = sample_len

        in_features =  sample_len*features*2 + tim_dim
        hidden_size = (in_features + emb_dim)*2//3
        self.fc_state = nn.Sequential(
            nn.Linear(in_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, emb_dim),
        )

        input_dim = tim_dim + (sample_len-1)*features*2
        hidden_size = (input_dim+emb_dim)*2//3
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
        self.mask_token = nn.Linear(in_features=sample_len*feature_dim, out_features=emb_dim)
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


class STALLM_MIMO(nn.Module):
    """
    支持物理分群（流、浪、風）的多輸入多輸出時空架構
    """
    def __init__(self, basemodel, sample_len, output_len,
                 input_dim, output_dim, node_emb_dim, sag_dim, sag_tokens,
                 node_embeddings=None, use_node_embedding=True,
                 use_timetoken=True, use_sandglassAttn=True,
                 dropout=0, trunc_k=16, t_dim=64): 
        super().__init__()
        
        self.sample_len = sample_len
        self.output_len = output_len
        self.emb_dim = basemodel.emb_dim
        self.basemodel = basemodel
        self.use_sandglassAttn = use_sandglassAttn
        self.use_node_embedding = use_node_embedding
        
        tim_dim = t_dim * 2
        
        # 1. 物理維度定義 (根據 clear.py 的 0,1 | 2,3,4,5 | 6,7 分佈)
        self.dims = {'flow': 2, 'wave': 4, 'wind': 2} 

        # 2. 為每一組物理量配置獨立的編碼器
        from model.model import Node2Token_Independent, TimeEmbedding, NodeEmbedding, DecodingLayer
        
        self.tokenizer_flow = Node2Token_Independent(sample_len, self.dims['flow'], node_emb_dim, self.emb_dim, tim_dim, dropout, use_node_embedding)
        self.tokenizer_wave = Node2Token_Independent(sample_len, self.dims['wave'], node_emb_dim, self.emb_dim, tim_dim, dropout, use_node_embedding)
        self.tokenizer_wind = Node2Token_Independent(sample_len, self.dims['wind'], node_emb_dim, self.emb_dim, tim_dim, dropout, use_node_embedding)
        
        # 3. 空間壓縮與物理對齊模塊 (以 Flow 為核心，Wind/Wave 為背景)
        if use_sandglassAttn:
            self.sag_flow = SAG(sag_dim, sag_tokens, self.emb_dim, sample_len, self.dims['flow'], dropout)
            self.sag_wave = SAG(sag_dim, sag_tokens, self.emb_dim, sample_len, self.dims['wave'], dropout)
            self.sag_wind = SAG(sag_dim, sag_tokens, self.emb_dim, sample_len, self.dims['wind'], dropout)
            self.cross_alignment = CrossModalityAlignment(self.emb_dim, num_heads=4, dropout=dropout)

        # 4. 並行多頭解碼器
        self.head_flow = DecodingLayer(self.emb_dim, self.dims['flow'] * output_len)
        self.head_wave = DecodingLayer(self.emb_dim, self.dims['wave'] * output_len)
        self.head_wind = DecodingLayer(self.emb_dim, self.dims['wind'] * output_len)

        # 基礎組件
        self.timeembedding = TimeEmbedding(t_dim=t_dim)
        if use_node_embedding:
            self.node_embd_layer = NodeEmbedding(node_embeddings, node_emb_dim, trunc_k, dropout)

    def forward(self, x, timestamp, prompt_prefix, mask):
        B, N, TF = x.shape 
        x_reshaped = x.view(B, N, self.sample_len, -1)
        mask_reshaped = mask.view(B, N, self.sample_len, -1)
        
        # 💡 物理索引精確切片 (对齐 8 维新数据)
        x_f = x_reshaped[..., [0, 1]].contiguous().view(B, N, -1)
        m_f = mask_reshaped[..., [0, 1]].contiguous().view(B, N, -1)
        
        # 波浪增加第 5 维 (包含 sin 和 cos)
        x_wa = x_reshaped[..., [2, 3, 4, 5]].contiguous().view(B, N, -1)
        m_wa = mask_reshaped[..., [2, 3, 4, 5]].contiguous().view(B, N, -1)
        
        # 海风被挤到了第 6, 7 维
        x_wi = x_reshaped[..., [6, 7]].contiguous().view(B, N, -1)
        m_wi = mask_reshaped[..., [6, 7]].contiguous().view(B, N, -1)

        te = self.timeembedding(timestamp[:, :self.sample_len, :])
        ne = self.node_embd_layer() if self.use_node_embedding else None

        # 獨立編碼
        tokens_f = self.tokenizer_flow(x_f, te, ne, m_f)
        tokens_wa = self.tokenizer_wave(x_wa, te, ne, m_wa)
        tokens_wi = self.tokenizer_wind(x_wi, te, ne, m_wi)

        # 物理對齊
        if self.use_sandglassAttn:
            s_f, _ = self.sag_flow.encode(tokens_f)
            s_wa, _ = self.sag_wave.encode(tokens_wa)
            s_wi, _ = self.sag_wind.encode(tokens_wi)
            
            # 融合海浪與海風的背景動力學特徵
            context = (s_wa + s_wi) / 2
            aligned = self.cross_alignment(core_state=s_f, context_state=context)
        else:
            aligned = tokens_f
            
        # LLM 推理
        hidden = self.basemodel(aligned)
        
        # 解碼
        if self.use_sandglassAttn:
            decoded = self.sag_flow.decode(hidden, tokens_f)
        else:
            decoded = hidden
            
        decoded += tokens_f # 殘差
        
        return {
            "flow": self.head_flow(decoded),
            "wave": self.head_wave(decoded),
            "wind": self.head_wind(decoded)
        }, []

    def params_num(self):
        total = sum(p.numel() for p in self.parameters()) + sum(p.numel() for p in self.buffers())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return total, trainable

    def grad_state_dict(self):
        return {n: p.detach() for n, p in self.named_parameters() if p.requires_grad}

    def save(self, path:str): torch.save(self.grad_state_dict(), path)
    def load(self, path:str): self.load_state_dict(torch.load(path), strict=False)