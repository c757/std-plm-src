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
    全新升级的多输入多输出（MIMO）时空大语言模型架构
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
        self.sag_tokens = sag_tokens
        
        tim_dim = t_dim * 2

        # 1. 物理要素切片定义
        self.dims = {'flow': 2, 'wind': 2, 'wave': 1, 'tide': 2} 

        # 2. 通道独立编码器 (Channel-Independent Tokenizers)
        self.tokenizer_flow = Node2Token_Independent(sample_len, self.dims['flow'], node_emb_dim, self.emb_dim, tim_dim, dropout, use_node_embedding)
        self.tokenizer_wind = Node2Token_Independent(sample_len, self.dims['wind'], node_emb_dim, self.emb_dim, tim_dim, dropout, use_node_embedding)
        
        # 3. 独立的空间压缩沙漏与跨模态对齐
        if use_sandglassAttn:
            self.sag_flow = SAG(sag_dim, sag_tokens, self.emb_dim, sample_len, self.dims['flow'], dropout)
            self.sag_wind = SAG(sag_dim, sag_tokens, self.emb_dim, sample_len, self.dims['wind'], dropout)
            self.cross_alignment = CrossModalityAlignment(self.emb_dim, num_heads=4, dropout=dropout)

        # 4. 并行多头解码器
        self.head_flow = DecodingLayer(self.emb_dim, self.dims['flow'] * output_len)
        self.head_wind = DecodingLayer(self.emb_dim, self.dims['wind'] * output_len)

        # ==========================================
        # 5. 补全缺失的基础组件初始化（修复报错的核心）
        # ==========================================
        self.timeembedding = TimeEmbedding(t_dim=t_dim)

        if use_node_embedding:
            self.node_embd_layer = NodeEmbedding(node_embeddings=node_embeddings, node_emb_dim=node_emb_dim, k=trunc_k, dropout=dropout)

        self.layer_norm = nn.LayerNorm(self.emb_dim)

    def forward(self, x:torch.FloatTensor, timestamp:torch.Tensor, prompt_prefix:Optional[torch.LongTensor], mask:torch.LongTensor):
        B, N, TF = x.shape 
        
        # 1. 解耦输入特征：将 F 维重新剥离出来
        x_reshaped = x.view(B, N, self.sample_len, -1)
        mask_reshaped = mask.view(B, N, self.sample_len, -1)
        
        # 切片提取特征 (保证各个通道数据的纯净)
        x_flow = x_reshaped[..., 0:2].contiguous().view(B, N, -1)
        m_flow = mask_reshaped[..., 0:2].contiguous().view(B, N, -1)
        
        x_wind = x_reshaped[..., 2:4].contiguous().view(B, N, -1)
        m_wind = mask_reshaped[..., 2:4].contiguous().view(B, N, -1)

        # 获取时间与节点 Embedding
        te = self.timeembedding(timestamp[:, :self.sample_len, :])
        ne = self.node_embd_layer() if self.use_node_embedding else None

        # 2. 独立编码
        spatial_token_flow = self.tokenizer_flow(x_flow, te, ne, m_flow)
        spatial_token_wind = self.tokenizer_wind(x_wind, te, ne, m_wind)

        # 3. 空间压缩与跨模态对齐 (TimeCMA 核心逻辑)
        if self.use_sandglassAttn:
            # 分别压缩至 128 个超节点
            s_state_flow, _ = self.sag_flow.encode(spatial_token_flow)
            s_state_wind, _ = self.sag_wind.encode(spatial_token_wind)
            
            # 【物理对齐】：以洋流为 Query，以海风为 Key/Value 提取驱动特征
            aligned_state = self.cross_alignment(core_state=s_state_flow, context_state=s_state_wind)
        else:
            aligned_state = spatial_token_flow # 降级方案
            
        # 4. 送入大语言模型 (LLM) 进行高维时间逻辑推演
        # 这里只送入了对齐后的融合状态，极大地节省了序列长度和显存
        hidden_state = self.basemodel(aligned_state)
        
        # 5. 超节点解码回 N 个物理网格点
        if self.use_sandglassAttn:
            decoded_state = self.sag_flow.decode(hidden_state, spatial_token_flow)
        else:
            decoded_state = hidden_state
            
        decoded_state += spatial_token_flow # 残差
        
        # 6. 多头并行输出预测结果
        # 这样网络在反向传播时，风和流的梯度不会相互干扰
        pred_flow = self.head_flow(decoded_state)
        # pred_wind = self.head_wind(decoded_state)

        # 返回一个字典，支持 MIMO 损失函数的独立计算
        return {"flow": pred_flow}, []
    def grad_state_dict(self):
        params_to_save = filter(lambda p: p[1].requires_grad, self.named_parameters())
        save_list = [p[0] for p in params_to_save]
        return {name: param.detach() for name, param in self.state_dict().items() if name in save_list}
        
    def save(self, path:str):
        selected_state_dict = self.grad_state_dict()
        torch.save(selected_state_dict, path)
    
    def load(self, path:str):
        loaded_params = torch.load(path)
        self.load_state_dict(loaded_params, strict=False)
    
    def params_num(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_params += sum(p.numel() for p in self.buffers())
        
        total_trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return total_params, total_trainable_params
