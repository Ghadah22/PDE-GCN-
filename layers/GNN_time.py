# layers/GNN_time.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from layers.Transformer_encoder import TransformerEncoder
from torchdiffeq import odeint_adjoint as odeint

# ---------- utils for patch graph ----------
def corr_over_patches(patch_tokens: torch.Tensor) -> torch.Tensor:
    """
    patch_tokens: [B, N, P, D]  (per-variable per-patch embeddings)
    return corr per variable: [B, N, P, P]
    """
    B, N, P, D = patch_tokens.shape
    x = patch_tokens - patch_tokens.mean(dim=-1, keepdim=True)
    x = x / (patch_tokens.std(dim=-1, keepdim=True) + 1e-6)
    # inner-product over D
    corr = torch.einsum('bnpd,bnqd->bnpq', x, x) / (D - 1 + 1e-6)
    return corr

def topk_A_from_corr(corr: torch.Tensor, k: int) -> torch.Tensor:
    # corr: [B,N,P,P]
    B, N, P, _ = corr.shape
    idx = torch.topk(corr, k=min(k+1, P), dim=-1).indices
    mask = torch.zeros_like(corr)
    mask.scatter_(-1, idx, 1.0)
    A = corr * mask
    A = 0.5 * (A + A.transpose(-1, -2))
    A = A + torch.eye(P, device=A.device).view(1,1,P,P)
    deg = A.sum(-1).clamp_min(1e-6)
    Dinv = deg.pow(-0.5).unsqueeze(-1)
    A_norm = Dinv * A * Dinv.transpose(-1, -2)
    return A_norm  # [B,N,P,P]

# ------------- ODE across patches ----------------
class _PatchRHS(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.theta = nn.Linear(d_model, d_model, bias=False)
        self.alpha_raw = nn.Parameter(torch.tensor(0.1))
        self.A = None  # [B,N,P,P]
        self.L = None  # [B,N,P,P]

    def set_graph(self, A_norm):
        self.A = A_norm
        B, N, P, _ = A_norm.shape
        I = torch.eye(P, device=A_norm.device).view(1,1,P,P)
        self.L = I - A_norm

    def forward(self, t, x):
        # x: [B,N,P,D]
        alpha = F.softplus(self.alpha_raw).clamp(max=2.0)
        diff  = alpha * torch.einsum('bnpr,bnrd->bnpd', self.L, x)
        react = self.theta(torch.einsum('bnpr,bnrd->bnpd', self.A, x))
        return diff + react  # [B,N,P,D]

class _PatchPDE(nn.Module):
    def __init__(self, d_model: int, k: int, t_span=(0.0, 0.2), solver="implicit_adams", rtol=1e-4, atol=1e-6):
        super().__init__()
        self.k = k
        self.rhs = _PatchRHS(d_model)
        self.register_buffer("tspan", torch.tensor(list(t_span), dtype=torch.float32))
        self.solver = solver; self.rtol = rtol; self.atol = atol

    def forward(self, patch_tokens: torch.Tensor):
        """
        patch_tokens: [B,N,P,D]
        """
        with torch.no_grad():
            corr = corr_over_patches(patch_tokens)        # [B,N,P,P]
            A = topk_A_from_corr(corr, self.k)            # [B,N,P,P]
        self.rhs.set_graph(A)
        t = self.tspan.to(patch_tokens.device)
        #y = odeint(self.rhs, patch_tokens, t, method=self.solver, rtol=self.rtol, atol=self.atol)[-1]
        
        # FIX: 'atch_tokens' -> 'patch_tokens'
        y = odeint(self.rhs, patch_tokens, t,method=self.solver)[-1]
        return F.relu(y)
                                         # [B,N,P,D]

# ----------- GCN over a single global patch graph ----------
class _GCNPatchBlock(nn.Module):
    def __init__(self, d_model, dropout, n_heads, d_ff, num_layers, activation):
        super().__init__()
        self.conv1 = GCNConv(d_model, d_model)
        self.conv2 = GCNConv(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.transformer = TransformerEncoder(d_model, n_heads, num_layers, d_ff, dropout)
        self.act = nn.Sigmoid() if activation == 'sigmoid' else nn.ReLU()

    def forward(self, patch_tokens, edge_index, B, N, P, D):
        # Build PyG input: [B*N*P, D]
        x_nodes = patch_tokens.reshape(B*N, P, D)  # one graph per (B*N), but we’ll reuse a shared EI via Batch
        data_list = [Data(x=x_nodes[i], edge_index=edge_index) for i in range(B*N)]
        batch = Batch.from_data_list(data_list)    # merges graphs
        h = self.act(self.conv1(batch.x, batch.edge_index))
        h = self.dropout(h)
        h = self.act(self.conv2(h, batch.edge_index))
        h = self.dropout(h)
        h = h.reshape(B, N, P, D)
        #h = self.transformer(x_nodes=h.reshape(B*N, P, D), context=None, mask=None).reshape(B, N, P, D)
        # self-attention over P patches per (B,N)
        h_bn = h.reshape(B * N, P, D)            # [B*N, P, D]
        h = self.transformer(h_bn, h_bn, None)   # (query, value, mask)
        h = h.reshape(B, N, P, D)

        return h

# ---------------- main module -----------------------------
class MultiLayerGCN_time(nn.Module):
    """
    Input:
      tokens:     [B,N,D]          (variable tokens, unused here but kept for API compatibility)
      x_bn:       [B,N,L]          (raw series)
      patch_tok:  [B,N,P,D]        (per-variable per-patch tokens)
    Output:
      [B,N,D]     (pooled over patches)
    """
    def __init__(self, num_layers, d_model, dropout, n_heads, d_ff, k, activation):
        super().__init__()
        self.d_model = d_model
        self.k = k
        self.gcn = _GCNPatchBlock(d_model, dropout, n_heads, d_ff, num_layers=1, activation=activation)
        self.pde = _PatchPDE(d_model, k=self.k)
        self.pool = nn.AdaptiveAvgPool1d(1)  # pool over P

    @torch.no_grad()
    def _edge_index_P(self, P: int, device: torch.device):
        # simple ring or fully-connected top-k could be here; use ring+diagonal as stable EI
        rows = torch.arange(P, device=device)
        cols = (rows + 1) % P
        ei = torch.stack([torch.cat([rows, rows]), torch.cat([cols, rows])], dim=0)  # ring + self
        return ei.long()

    def forward(self, tokens: torch.Tensor, x_bn: torch.Tensor, patch_tok: torch.Tensor):
        """
        tokens:   [B,N,D]   (ignored)
        x_bn:     [B,N,L]   (ignored here; graph built from patch_tok)
        patch_tok:[B,N,P,D]
        """
        B, N, P, D = patch_tok.shape
        # 1) build patch graph and run GCN
        edge_index = self._edge_index_P(P, patch_tok.device)  # [2, E]
        h = self.gcn(patch_tok, edge_index, B, N, P, D)       # [B, N, P, D]
    
        # 2) PDE refinement across patches
        h = self.pde(h)                                       # [B, N, P, D]
    
        # 3) pool over patches -> [B, N, D]
        h = h.mean(dim=2) 
        return h
