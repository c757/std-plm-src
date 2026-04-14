from typing import Iterator, Mapping
import torch
import torch.nn as nn
from typing import Any, Dict, Optional, Tuple, Union
from utils.utils import norm_Adj, lap_eig, topological_sort
from model.sandglassAttn import SAG
import numpy as np
from model.position import PositionalEncoding


class DecodingLayer(nn.Module):

    def __init__(self, input_dim ,emb_dim, output_dim):
        super().__init__()

        hidden_size = (emb_dim+output_dim)*2//3
        self.fc = nn.Sequential(
            nn.Linear(emb_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_dim),
        )
        #nn.Linear(in_features=input_dim,out_features=output_dim)
        
    def forward(self, llm_hidden):

        out = self.fc(llm_hidden)

        return out

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

    def forward(self,x,te,mask):
        # te(B,T,tim_dim)

        B,N,TF = x.shape

        x = x.view(B,N,self.sample_len,-1) #B,N,T,F
        x = torch.concat((x,mask.view(B,N,self.sample_len,-1)),dim=-1)
        x = x.mean(dim=1) #B,T,F

        state = x.view(B,1,-1)
        state = torch.concat((state,te[:,-1:,:]),dim=-1)#(B,1,TF+tim_dim)
        state = self.fc_state(state)

        grad = (x[:,1:,:] - x[:,:-1,:]).view(B,1,-1)#(B,1,(T-1)F)
        grad = torch.concat((grad,te[:,-1:,:]),dim=-1)#(B,1,(T-1)F+tim_dim)
        grad = self.fc_grad(grad)

        out = torch.concat((state,grad),dim=1)

        out = self.ln(out)

        return out


class Node2Token(nn.Module):
    def __init__(self,sample_len, features, node_emb_dim, emb_dim, tim_dim, dropout, use_node_embedding):
        super().__init__()

        in_features = sample_len*features*2
        
        self.use_node_embedding = use_node_embedding

        state_features =  tim_dim
        if use_node_embedding:
            state_features += node_emb_dim

        self.fc1 = nn.Sequential(
            nn.Linear(in_features, emb_dim),
        )
        
        hidden_size = node_emb_dim
        self.state_fc = nn.Sequential(
            nn.Linear(state_features, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,emb_dim),
        )

        self.mask_token = nn.Linear(in_features=sample_len*features,out_features=emb_dim)

        self.ln = nn.LayerNorm(emb_dim)

    def forward(self,x,te,ne,mask):
        #te(B,T,tim_dim)  ne(N,node_emb_dim) mask(B,T,N,F)
        B,N,TF = x.shape

        mask = mask.permute(0,2,1,3).contiguous().view(B,N,-1) #(B,N,TF)
        x = torch.concat((x,mask),dim=-1)

        state = te[:,-1:,:].repeat(1,N,1)
        if self.use_node_embedding:
            ne = torch.unsqueeze(ne,dim=0).repeat(B,1,1)
            state = torch.concat((state,ne),dim=-1)
        state = self.state_fc(state)

        x = self.fc1(x) #B,N,D
        x += self.mask_token(mask) #test

        out = state + x

        out = self.ln(out)

        return out


class STALLM(nn.Module):
    # 将 adj_mx, dis_mx 替换为 node_embeddings
    def __init__(self, basemodel, sample_len, output_len,
                 input_dim, output_dim, node_emb_dim, sag_dim, sag_tokens,
                 node_embeddings=None, use_node_embedding=True,
                 use_timetoken=True, use_sandglassAttn=True,
                 dropout=0, trunc_k=16, t_dim=64, wo_conloss=True): # 强制 wo_conloss 默认为 True
        super().__init__()

        # 🚨 必须关闭：6万节点无法进行拓扑排序
        self.topological_sort_node = False

        tim_dim = t_dim * 2 # hour, week

        # 🚨 删除了原有的 self.setadj(adj_mx, dis_mx)

        self.output_dim = output_dim
        self.input_dim = input_dim
        self.emb_dim = basemodel.emb_dim
        self.basemodel = basemodel
        self.sample_len = sample_len
        self.output_len = output_len
        self.sag_tokens = sag_tokens

        self.use_sandglassAttn = use_sandglassAttn
        if use_sandglassAttn:
            # 🚨 必须关闭：约束损失计算涉及 63200x63200 矩阵乘法，会导致显存瞬间溢出
            self.wo_conloss = True
            self.sandglassAttn = SAG(sag_dim=sag_dim, sag_tokens=sag_tokens, emb_dim=self.emb_dim, sample_len=sample_len, features=input_dim, dropout=dropout)

        self.spatialTokenizer = Node2Token(sample_len=sample_len, features=input_dim, node_emb_dim=node_emb_dim,
                                            emb_dim=self.emb_dim,
                                            tim_dim=tim_dim, dropout=dropout, use_node_embedding=use_node_embedding)

        self.out_mlp = DecodingLayer(input_dim=output_dim*sample_len,
                                     emb_dim=self.emb_dim,
                                     output_dim=output_dim*output_len)

        self.timeembedding = TimeEmbedding(t_dim=t_dim)

        self.use_node_embedding = use_node_embedding
        if use_node_embedding:
            # 🚨 改为传入预先计算好的 node_embeddings
            self.node_embd_layer = NodeEmbedding(node_embeddings=node_embeddings, node_emb_dim=node_emb_dim, k=trunc_k, dropout=dropout)

        self.use_timetoken = use_timetoken
        if use_timetoken:
            self.timeTokenizer = Time2Token(sample_len=sample_len, features=input_dim,
                                            emb_dim=self.emb_dim,
                                            tim_dim=tim_dim, dropout=dropout)
        
        self.layer_norm = nn.LayerNorm(self.emb_dim)

    # 往下保留原有的 forward、save 等方法...
    # 🚨 滑到最后，把 def setadj(self, adj_mx, dis_mx): 整个函数彻底删掉！

    def forward(self,x:torch.FloatTensor,timestamp:torch.Tensor,prompt_prefix:Optional[torch.LongTensor],mask:torch.LongTensor):
        other_loss = []

        # timestamp (B,T,4)
        timestamp = timestamp[:,:self.sample_len,:]

        B,N,TF = x.shape #(Batch,N,T*features)
        # emb of time
        te = self.timeembedding(timestamp) #(B,T,tim_dim)
        # emb of nodes
        if self.use_node_embedding:
            ne = self.node_embd_layer()
        else:
            ne = None

        # spatial token
        spatial_token = self.spatialTokenizer(x,te,ne,mask)
        if self.topological_sort_node:
            spatial_token = spatial_token[:,self.node_order,:]

        # spatial -> sandglassAttn
        st_embedding = spatial_token
        s_num = N
        if self.use_sandglassAttn:
            s_num = self.sag_tokens
            st_embedding,attn_weights = self.sandglassAttn.encode(st_embedding) #(B,N',D) #attn_weights(B,N',N)
            if not self.wo_conloss:
                scale = attn_weights.sum(dim=1)#(B,N)

                sag_score = torch.einsum('bmn,bhn->bhm',self.adj_mx[None,:,:],attn_weights)
                other_loss.append(-((sag_score*attn_weights-attn_weights*attn_weights)).sum(dim=2).mean()*10)

                Dirichlet = torch.distributions.dirichlet.Dirichlet(self.alpha)
                other_loss.append(-Dirichlet.log_prob(torch.softmax(scale,dim=-1)).sum())

        if self.use_timetoken:
            time_tokens = self.timeTokenizer(x,te,mask)
            time_tokens_idx = st_embedding.shape[1]
            st_embedding = torch.concat([time_tokens,st_embedding],dim=1)

        if prompt_prefix is not None:
            prompt_len,_ = prompt_prefix.shape
            prompt_embedding = self.basemodel.getembedding(prompt_prefix).view(1,prompt_len,-1)
            prompt_embedding = prompt_embedding.repeat(B,1,1)
            st_embedding = torch.concat([prompt_embedding,st_embedding],dim=1)

        hidden_state = st_embedding

        hidden_state = self.basemodel(hidden_state)
        s_state = hidden_state[:,-s_num:,:]


        if self.use_sandglassAttn:
            s_state = self.sandglassAttn.decode(s_state,spatial_token) 
        s_state += spatial_token

        if self.topological_sort_node:
            s_state = s_state[:,self.node_order_rev,:]

        if self.use_timetoken:
            t_state = hidden_state[:,-time_tokens_idx-1:-time_tokens_idx,:]
            t_state += time_tokens[:,-1:,:]

            s_state += t_state

        s_state = self.layer_norm(s_state)

        out = self.out_mlp(s_state)

        return out, other_loss

    def grad_state_dict(self):
        params_to_save = filter(lambda p: p[1].requires_grad, self.named_parameters())
        save_list = [p[0] for p in params_to_save]
        return  {name: param.detach() for name, param in self.state_dict().items() if name in save_list}
        
    
    def save(self, path:str):
        
        selected_state_dict = self.grad_state_dict()
        torch.save(selected_state_dict, path)
    
    def load(self, path:str):

        loaded_params = torch.load(path)
        self.load_state_dict(loaded_params,strict=False)
    
    def params_num(self):
        total_params = sum(p.numel() for p in self.parameters())
        total_params += sum(p.numel() for p in self.buffers())
        
        total_trainable_params = sum(
            p.numel() for p in self.parameters() if p.requires_grad)
        
        return total_params, total_trainable_params

    

# if __name__ == "__main__":
    
    # model = Phi4ST_GNN(8)

    # data = torch.rand(1,12,8)

    # print(model(x=data))

    # optimizer = torch.optim.Adam(params=filter(lambda x: x.requires_grad ,model.parameters()),lr=1e-3)

    # model.save('test.pth')

    # model.load('test.pth')


