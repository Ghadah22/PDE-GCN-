# layers/text_fusion.py
import torch, torch.nn as nn, torch.nn.functional as F

class TextFusion(nn.Module):
    """
    Inputs:
      tokens:    [B, N, D]          (your per-variable tokens)
      ctx_seq:   [B, L, E]          (LLM sentence embeddings per timestamp)
    Returns:
      tokens_out [B, N, D]          (FiLM + cross-attn fused)
    """
    def __init__(self, d_model: int, ctx_dim: int, n_heads: int = 1, dropout: float = 0.1):
        super().__init__()
        self.proj_ctx   = nn.Linear(ctx_dim, d_model)
        self.gamma      = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        self.beta       = nn.Sequential(nn.Linear(d_model, d_model), nn.Tanh())
        # tiny cross-attn: tokens (Q) attend to time-context (K,V)
        self.q  = nn.Linear(d_model, d_model)
        self.k  = nn.Linear(d_model, d_model)
        self.v  = nn.Linear(d_model, d_model)
        self.o  = nn.Linear(d_model, d_model)
        self.h  = n_heads
        self.drop = nn.Dropout(dropout)

    def _mhsa(self, q, k, v):
        # q: [B,N,D], k/v: [B,L,D]
        B, N, D = q.shape
        L = k.size(1); H = self.h; d = D // H
        q = self.q(q).view(B, N, H, d).transpose(1,2)   # [B,H,N,d]
        k = self.k(k).view(B, L, H, d).transpose(1,2)   # [B,H,L,d]
        v = self.v(v).view(B, L, H, d).transpose(1,2)   # [B,H,L,d]
        att = torch.einsum('bhnc,bhlc->bhln', q, k) / (d**0.5)
        A = F.softmax(att, dim=-1)
        A = self.drop(A)
        out = torch.einsum('bhln,bhld->bhnd', A, v)     # [B,H,N,d]
        out = out.transpose(1,2).contiguous().view(B, N, D)
        return self.o(out)

    def forward(self, tokens, ctx_seq):
        # 1) pool context in time and FiLM-condition tokens
        ctx_seq = self.proj_ctx(ctx_seq)           # [B,L,D]
        g = ctx_seq.mean(dim=1)                    # [B,D]
        gamma = self.gamma(g).unsqueeze(1)         # [B,1,D]
        beta  = self.beta(g).unsqueeze(1)          # [B,1,D]
        tokens_film = (1.0 + gamma) * tokens + beta

        # 2) cross-attend tokens to the whole time-context
        ctx = ctx_seq                               # [B,L,D]
        tokens_att = self._mhsa(tokens_film, ctx, ctx)

        return F.relu(tokens_film + 0.5 * tokens_att)
