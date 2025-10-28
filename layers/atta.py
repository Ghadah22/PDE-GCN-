# layers/attn_tsa.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class FullAttention(nn.Module):
    """Scaled dot-product attention (batch-first, multihead packed outside)."""
    def __init__(self, scale=None, attention_dropout=0.1):
        super().__init__()
        self.scale = scale
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values):
        # queries/keys/values: [B, L, H, E]
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or (1.0 / math.sqrt(E))
        # scores: [B, H, L, S]
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        A = torch.softmax(scale * scores, dim=-1)
        A = self.dropout(A)
        # output: [B, L, H, D]
        V = torch.einsum("bhls,bshd->blhd", A, values)
        return V.contiguous()


class AttentionLayer(nn.Module):
    """Multi-head attention with linear projections (batch-first)."""
    def __init__(self, d_model, n_heads, d_keys=None, d_values=None, mix=True, dropout=0.1):
        super().__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.inner_attention = FullAttention(attention_dropout=dropout)
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection   = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection   = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix     = mix

    def forward(self, queries, keys, values):
        # queries/keys/values: [B, L, D]
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads
        q = self.query_projection(queries).view(B, L, H, -1)
        k = self.key_projection(keys).view(B, S, H, -1)
        v = self.value_projection(values).view(B, S, H, -1)
        out = self.inner_attention(q, k, v)             # [B, L, H, d_v]
        if self.mix:
            out = out.transpose(2, 1).contiguous()      # [B, H, L, d_v] (optional)
        out = out.view(B, L, -1)                        # [B, L, H*d_v]
        return self.out_projection(out)                 # [B, L, D]


class TwoStageAttentionLayer(nn.Module):
    """
    TSA over segments.
    Input/Output: [B, data_dim (ts_d), seg_num (L), D]
      - We apply MSA along seg_num per each data-dim (ts_d).
    """
    def __init__(self, seg_num, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.seg_num = seg_num
        self.time_attention = AttentionLayer(d_model, n_heads, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.mlp   = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        # x: [B, ts_d, seg_num, D]
        B = x.size(0)
        time_in  = rearrange(x, 'b ts seg d -> (b ts) seg d')               # [B*ts_d, seg_num, D]
        time_enc = self.time_attention(time_in, time_in, time_in)           # [B*ts_d, seg_num, D]
        time_out = self.norm1(time_in + self.dropout(time_enc))             # residual
        time_out = self.norm2(time_out + self.dropout(self.mlp(time_out)))  # FFN
        final_out = rearrange(time_out, '(b ts) seg d -> b ts seg d', b=B)  # [B, ts_d, seg_num, D]
        return final_out
