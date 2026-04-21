from typing import Iterator, Mapping
import torch
#torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
from modelscope.models import Model
from torch.nn.parameter import Parameter
from typing import Any, Dict, Optional, Tuple, Union
from swift.tuners import Swift, LoraConfig
from modelscope import AutoTokenizer as MS_AutoTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer as HF_AutoTokenizer

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self,x):
        raise NotImplementedError('error')
    
    def getembedding(self,x):
        raise NotImplementedError('error')
    
    def gettokenizer(self):
        raise NotImplementedError('error')
    
    def getmonthembedding(self):
        months = ['January','February','March','April','May','June','July','August','September','October','November','December']
        inputs = self.tokenizer.convert_tokens_to_ids(months)
        #self.tokenizer('January,February,March,April,May,June,July,August,September,October,November,December', 
                  #  return_tensors="pt", return_attention_mask=False)
        #month_ids= inputs['input_ids'].cuda().view(-1,1)[::2]
        month_ids = torch.tensor(inputs).cuda().view(-1,1)
        month_embedding = self.getembedding(month_ids).view(-1,self.emb_dim)
        return month_embedding
    
    def getweekembedding(self):
        weeks = ['Sunday','Monday','Tuesday','Wednesday','Thursday','Friday','Saturday']
        inputs = self.tokenizer.convert_tokens_to_ids(weeks)
        week_ids = torch.tensor(inputs).cuda().view(-1,1)
        week_embedding = self.getembedding(week_ids).view(-1,self.emb_dim)
        return week_embedding

class Phi2(BaseModel):
    def __init__(self,causal,lora,ln_grad,layers=None):
        super().__init__()

        causal = bool(causal)

        self.emb_dim = 2560

        llm = Model.from_pretrained('AI-ModelScope/phi-2',trust_remote_code=True)

        if not layers is None:

            llm.transformer.h = llm.transformer.h[:layers]

        for pblock in llm.transformer.h:
            mixer = pblock.mixer
            mixer.inner_attn.causal = causal
            mixer.inner_attn.causal = causal
        
        for name, param in llm.named_parameters():
            param.requires_grad_(False)

        if lora:

            lora_config = LoraConfig(
                    r=16,
                    target_modules=['Wqkv'],
                    lora_alpha=32,
                    lora_dropout=0.)
            
            llm = Swift.prepare_model(llm, lora_config,trust_remote_code=True)

        self.llm_embd = llm.transformer.embd # wte:51200->2560  (B,len,1) -> (B,len,emb_dim)

        self.llm_h = llm.transformer.h # ModuleList (B,len,emb_dim) ->  (B,len,emb_dim)
        
        if ln_grad:
            for i, (name, param) in enumerate(self.llm_h.named_parameters()):
                if 'ln' in name: # or 'mlp' in name:
                    param.requires_grad = True

        # ModelScope tokenizer for ModelScope-hosted model id
        try:
            self.tokenizer = MS_AutoTokenizer.from_pretrained("AI-ModelScope/phi-2", trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(
                "Failed to load ModelScope tokenizer for 'AI-ModelScope/phi-2'.\n"
                "This can happen when the model id isn't available on HuggingFace or ModelScope from this machine,\n"
                "or network/authentication is required.\n"
                "Possible fixes:\n"
                " - Ensure you have network access and the model id is correct.\n"
                " - If the model is private, authenticate with the provider (e.g., `hf auth login` for HuggingFace or follow ModelScope auth).\n"
                " - Alternatively place the tokenizer files locally and point to the local path.\n"
                f"Original error: {e}") from e

    def forward(self,x:torch.FloatTensor):

        hidden_state = x

        for layer in self.llm_h:
            hidden_state = layer(hidden_state)

        out = hidden_state

        return out

    def getembedding(self, x:torch.FloatTensor):

        return self.llm_embd(x)
    
    def gettokenizer(self):

        return self.tokenizer 
    
    def getmonthembedding(self):
        inputs = self.tokenizer('January,February,March,April,May,June,July,August,September,October,November,December', 
                    return_tensors="pt", return_attention_mask=False)
        month_ids= inputs['input_ids'].cuda().view(-1,1)[::2]
        month_embedding = self.getembedding(month_ids).view(-1,self.emb_dim)
        return month_embedding


class GPT2(BaseModel):
    def __init__(self,causal,lora,ln_grad,layers=None):
        super().__init__()

        causal = bool(causal)

        self.emb_dim = 768

        self.llm = Model.from_pretrained('AI-ModelScope/gpt2',trust_remote_code=True)

        if not layers is None:

            self.llm.transformer.h = self.llm.transformer.h[:layers]
        
        self.causal = causal

        for name, param in self.llm.named_parameters():
            param.requires_grad_(False)

        if lora:

            lora_config = LoraConfig(
                    r=16,
                    target_modules=['q_attn','c_attn'],
                    lora_alpha=32,
                    lora_dropout=0.)
            
            self.llm = Swift.prepare_model(self.llm, lora_config,trust_remote_code=True).model

        # self.llm_embd = llm.transformer.wte # wte:50257,1->51200,768  (B,len,1) -> (B,len,emb_dim) # wte:51200->2560  (B,len,1) -> (B,len,emb_dim)

        
        if ln_grad:
            for i, (name, param) in enumerate(self.llm.named_parameters()):
                if 'ln' in name  or 'wpe' in name:
                    param.requires_grad = True

        # ModelScope tokenizer for ModelScope-hosted model id
        try:
            self.tokenizer = MS_AutoTokenizer.from_pretrained("AI-ModelScope/gpt2", trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(
                "Failed to load ModelScope tokenizer for 'AI-ModelScope/gpt2'.\n"
                "See suggestions in the error message for phi2 above.\n"
                f"Original error: {e}") from e

    def forward(self,x:torch.FloatTensor,attention_mask=None):

        out = self.llm(inputs_embeds=x,attention_mask=attention_mask,output_hidden_states=True).hidden_states[-1]

        return out

    def getembedding(self, x:torch.FloatTensor):

        return self.llm.transformer.wte(x)
    
    def gettokenizer(self):

        return self.tokenizer 


class Transformer(BaseModel):
    def __init__(self,causal,lora,ln_grad,layers=None):
        super().__init__()


        self.emb_dim = 768

        encoder_layer = nn.TransformerEncoderLayer(d_model=self.emb_dim, nhead=12, batch_first=True)
        self.llm = nn.TransformerEncoder(encoder_layer=encoder_layer,num_layers=3)


    def forward(self,x:torch.FloatTensor,attention_mask=None):

        out = self.llm(x)

        return out

   

class LLAMA3(BaseModel):
    def __init__(self,causal,lora,ln_grad,layers=None):
        super().__init__()

        causal = bool(causal)

        self.emb_dim = 4096

        self.llm = Model.from_pretrained('LLM-Research/Meta-Llama-3-8B-Instruct',trust_remote_code=True)

        print(self.llm)

        if not layers is None:

            self.llm.model.layers = self.llm.model.layers[:layers]
        
        self.causal = causal

        for name, param in self.llm.named_parameters():
            param.requires_grad_(False)

        if lora:

            lora_config = LoraConfig(
                    r=16,
                    target_modules=['q_proj','k_proj','v_proj','o_proj'],
                    lora_alpha=32,
                    lora_dropout=0.)
            
            self.llm = Swift.prepare_model(self.llm, lora_config,trust_remote_code=True).model

        # self.llm_embd = llm.transformer.wte # wte:50257,1->51200,768  (B,len,1) -> (B,len,emb_dim) # wte:51200->2560  (B,len,1) -> (B,len,emb_dim)

        
        if ln_grad:
            for i, (name, param) in enumerate(self.llm.named_parameters()):
                if 'norm' in name  or 'wpe' in name:
                    param.requires_grad = True

        # ModelScope tokenizer for Llama3 model (ModelScope)
        try:
            self.tokenizer = MS_AutoTokenizer.from_pretrained("LLM-Research/Meta-Llama-3-8B-Instruct", trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(
                "Failed to load ModelScope tokenizer for 'LLM-Research/Meta-Llama-3-8B-Instruct'.\n"
                "See suggestions in the error message for phi2 above.\n"
                f"Original error: {e}") from e

    def forward(self,x:torch.FloatTensor,attention_mask=None):

        out = self.llm(inputs_embeds=x,attention_mask=attention_mask,output_hidden_states=True).hidden_states[-1]

        return out

    def getembedding(self, x:torch.FloatTensor):

        return self.llm.model.embed_tokens(x)
    
    def gettokenizer(self):

        return self.tokenizer 


class QWEN(BaseModel):
    def __init__(self, causal, lora, ln_grad, layers=None):
        super().__init__()

        causal = bool(causal)

        # 准确的 HuggingFace 模型 ID
        model_id = 'Qwen/Qwen3-4B-Instruct-2507' 
        
        # 核心修改：使用 transformers 直接从 HuggingFace 加载
        # 强制使用 FP16 半精度加载，将模型体积直接砍半
        self.llm = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype=torch.float16)
        print(self.llm)

        # 动态读取特征维度
        self.emb_dim = self.llm.config.hidden_size

        if not layers is None:
            self.llm.model.layers = self.llm.model.layers[:layers]
        
        self.causal = causal

        for name, param in self.llm.named_parameters():
            param.requires_grad_(False)

        if lora:
            lora_config = LoraConfig(
                    r=16,
                    target_modules=['q_proj','k_proj','v_proj','o_proj'],
                    lora_alpha=32,
                    lora_dropout=0.)
            
            self.llm = Swift.prepare_model(self.llm, lora_config, trust_remote_code=True).model

        if ln_grad:
            for i, (name, param) in enumerate(self.llm.named_parameters()):
                if 'norm' in name:
                    param.requires_grad = True

        # QWEN is hosted on HuggingFace; use HF AutoTokenizer
        try:
            self.tokenizer = HF_AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load HuggingFace tokenizer for '{model_id}'.\n"
                "If this is a private HF repo, run `huggingface-cli login` or set environment token.\n"
                "If offline, provide a local path to tokenizer files.\n"
                f"Original error: {e}") from e

    def forward(self, x:torch.FloatTensor, attention_mask=None):
        # 1. 强制将输入数据转换为与大模型权重一致的半精度 (FP16)
        x = x.to(self.llm.dtype)
        
        out = self.llm(inputs_embeds=x, attention_mask=attention_mask, output_hidden_states=True).hidden_states[-1]
        
        # 2. 输出时将其转回单精度 (FP32)，防止后续的并行解码器 (DecodingLayer) 报错
        return out.to(torch.float32)

    def getembedding(self, x:torch.FloatTensor):
        return self.llm.model.embed_tokens(x)
    
    def gettokenizer(self):
        return self.tokenizer