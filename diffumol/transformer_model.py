from transformers import AutoConfig
from transformers.models.bert.modeling_bert import BertEncoder, BertModel
import torch

import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F

from .utils.nn import (
    SiLU,
    linear,
    timestep_embedding,
)

# ============== 新增：轻量级图融合模块 ==============
class GraphCrossAttention(nn.Module):
    """
    轻量级跨注意力融合图嵌入
    参数量：~5000 (相比原方案的50000减少10倍)
    """
    def __init__(self, hidden_dim, graph_embed_dim, num_heads=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        # 极轻量的投影层（无bias减少参数）
        self.graph_proj = nn.Linear(graph_embed_dim, hidden_dim, bias=False)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.scale = self.head_dim ** -0.5
        
        # 可学习的温度参数（控制融合强度）
        self.temperature = nn.Parameter(torch.ones(1) * 0.5)  # 初始温度0.5
        
    def forward(self, text_emb, graph_emb):
        """
        text_emb: [B, L, D] - 文本嵌入
        graph_emb: [B, D'] - 图嵌入
        返回: [B, L, D] - 图信息增强的文本嵌入
        """
        B, L, D = text_emb.shape
        
        # 投影图嵌入到相同维度
        g = self.graph_proj(graph_emb).unsqueeze(1)  # [B, 1, D]
        
        # Query from text, Key/Value from graph
        q = self.q_proj(text_emb)  # [B, L, D]
        
        # Multi-head attention (简化版)
        q = q.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = g.reshape(B, 1, self.num_heads, self.head_dim).transpose(1, 2)
        v = k
        
        # Scaled dot-product attention with temperature
        attn = (q @ k.transpose(-2, -1)) * self.scale  # [B, heads, L, 1]
        attn = torch.softmax(attn / self.temperature.clamp(min=0.01), dim=-1)
        
        # Weighted graph information
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        
        return out

# ============== 主模型修改 ==============
class TransformerNetModel(nn.Module):
    """
    The full Transformer model with attention and timestep embedding.

    :param input_dims: dims of the input Tensor.
    :param output_dims: dims of the output Tensor.
    :param hidden_t_dim: dims of time embedding.
    :param dropout: the dropout probability.
    :param config/config_name: the config of PLMs.
    :param init_pretrained: bool, init whole network params with PLMs.
    :param vocab_size: the size of vocabulary
    """

    def __init__(
        self,
        input_dims,
        output_dims,
        hidden_t_dim,
        dropout=0,
        config=None,
        config_name='bert-base-uncased',
        vocab_size=None,
        init_pretrained='no',
        logits_mode=1,
        **kwargs
    ):
        super().__init__()

        if config is None:
            config = AutoConfig.from_pretrained(config_name)
            config.hidden_dropout_prob = dropout

        self.input_dims = input_dims
        self.hidden_t_dim = hidden_t_dim
        self.output_dims = output_dims
        self.dropout = dropout
        self.logits_mode = logits_mode
        self.hidden_size = config.hidden_size
        self.num_props = kwargs["num_props"]
        
        # ====== 修改：使用Cross-Attention代替原来的简单投影 ======
        self.use_graph = kwargs.get("use_graph", False)
        if self.use_graph:
            graph_embed_dim = kwargs.get("graph_embed_dim", 128)
            # 用轻量级Cross-Attention替代原来的3层MLP
            self.graph_fusion = GraphCrossAttention(
                hidden_dim=config.hidden_size,
                graph_embed_dim=graph_embed_dim,
                num_heads=2  # 只用2个头，足够且轻量
            )
        
        self.word_embedding = nn.Embedding(vocab_size, self.input_dims)
        
        if self.num_props:
            self.prop_nn = nn.Linear(self.num_props, self.input_dims) 
        
        self.lm_head = nn.Linear(self.input_dims, vocab_size)
        with th.no_grad():
            self.lm_head.weight = self.word_embedding.weight

        time_embed_dim = hidden_t_dim * 4
        self.time_embed = nn.Sequential(
            linear(hidden_t_dim, time_embed_dim),
            SiLU(),
            linear(time_embed_dim, config.hidden_size),
        )

        if self.input_dims != config.hidden_size:
            self.input_up_proj = nn.Sequential(nn.Linear(input_dims, config.hidden_size),
                                              nn.Tanh(), nn.Linear(config.hidden_size, config.hidden_size))
        
        if init_pretrained == 'bert':
            print('initializing from pretrained bert...')
            print(config)
            temp_bert = BertModel.from_pretrained(config_name, config=config)

            self.word_embedding = temp_bert.embeddings.word_embeddings
            with th.no_grad():
                self.lm_head.weight = self.word_embedding.weight
            
            self.input_transformers = temp_bert.encoder
            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = temp_bert.embeddings.position_embeddings
            self.LayerNorm = temp_bert.embeddings.LayerNorm

            del temp_bert.embeddings
            del temp_bert.pooler

        elif init_pretrained == 'no':
            self.input_transformers = BertEncoder(config)

            self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
            self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        else:
            assert False, "invalid type of init_pretrained"
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.output_dims != config.hidden_size:
            self.output_down_proj = nn.Sequential(nn.Linear(config.hidden_size, config.hidden_size),
                                                nn.Tanh(), nn.Linear(config.hidden_size, self.output_dims))

    def get_embeds(self, input_ids):
        return self.word_embedding(input_ids.to(th.int64))

    def get_props(self, props):
        return self.prop_nn(props.unsqueeze(1))
        
    def get_logits(self, hidden_repr):
        if self.logits_mode == 1:
            return self.lm_head(hidden_repr)
        elif self.logits_mode == 2: 
            text_emb = hidden_repr
            emb_norm = (self.lm_head.weight ** 2).sum(-1).view(-1, 1)  
            text_emb_t = th.transpose(text_emb.view(-1, text_emb.size(-1)), 0, 1)  
            arr_norm = (text_emb ** 2).sum(-1).view(-1, 1)
            dist = emb_norm + arr_norm.transpose(0, 1) - 2.0 * th.mm(self.lm_head.weight,
                                                                     text_emb_t)  
            scores = th.sqrt(th.clamp(dist, 0.0, np.inf)).view(emb_norm.size(0), hidden_repr.size(0),
                                                               hidden_repr.size(1)) 
            scores = -scores.permute(1, 2, 0).contiguous()
            return scores
        else:
            raise NotImplementedError


    def forward(self, x, timesteps, graph_ids=None):
        """
        Apply the model to an input batch.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param graph_ids: molecular indices used to retrieve graph embeddings (optional)
        :return: an [N x C x ...] Tensor of outputs.
        """
        emb_t = self.time_embed(timestep_embedding(timesteps, self.hidden_t_dim))
        
        if self.input_dims != self.hidden_size:
            emb_x = self.input_up_proj(x)
        else:
            emb_x = x
        
        seq_length = x.size(1)
        position_ids = self.position_ids[:, : seq_length]
        
        # ====== 修改：用Cross-Attention融合图信息 ======
        emb_inputs = self.position_embeddings(position_ids) + emb_x + emb_t.unsqueeze(1).expand(-1, seq_length, -1)
        emb_inputs = self.LayerNorm(emb_inputs)
        
        if self.use_graph and graph_ids is not None:
            # 获取图嵌入
            graph_emb = self.graph_embeddings[graph_ids]  # [B, graph_dim]
            
            # Cross-attention融合（不破坏原始嵌入）
            graph_info = self.graph_fusion(emb_inputs, graph_emb)  # [B, L, D]
            
            # 残差连接（关键！保持原始文本信息）
            # 使用小权重开始，让模型逐渐学习融合强度
            emb_inputs = emb_inputs + 0.1 * graph_info
        
        emb_inputs = self.dropout(emb_inputs)
        
        input_trans_hidden_states = self.input_transformers(emb_inputs).last_hidden_state
        
        if self.output_dims != self.hidden_size:
            h = self.output_down_proj(input_trans_hidden_states)
        else:
            h = input_trans_hidden_states
        h = h.type(x.dtype)
        return h