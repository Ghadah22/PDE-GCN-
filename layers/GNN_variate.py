# layers/GNN_variate.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, Batch
from layers.Transformer_encoder import TransformerEncoder
from torchdiffeq import odeint_adjoint as odeint

# ---------------- utils: correlation graph ----------------
def pearson_corr(x_bn: torch.Tensor) -> torch.Tensor:
    """
    x_bn: [B, N, L]  (variables-first)
    return: corr [B, N, N]
    """
    B, N, L = x_bn.shape
    x = x_bn - x_bn.mean(dim=-1, keepdim=True)
    x = x / (x.std(dim=-1, keepdim=True) + 1e-6)
    return torch.matmul(x, x.transpose(-1, -2)) / (L - 1)  # [B,N,N]

def build_topk_A(corr: torch.Tensor, k: int) -> torch.Tensor:
    """
    corr: [B, N, N] (similarity), symmetric-ish
    returns A_norm (symmetric normalized) [B,N,N]
    """
    B, N, _ = corr.shape
    # keep top-k per row (excluding self), add self-loops
    idx = torch.topk(corr, k=k+1, dim=-1).indices  # top k+1 includes self
    mask = torch.zeros_like(corr)
    mask.scatter_(-1, idx, 1.0)
    A = corr * mask
    A = (A + A.transpose(-1, -2)) * 0.5
    # add I and normalize
    A = A + torch.eye(N, device=A.device).unsqueeze(0).expand(B, -1, -1)
    deg = A.sum(-1).clamp_min(1e-6)
    Dinv = deg.pow(-0.5).unsqueeze(-1)
    A_norm = Dinv * A * Dinv.transpose(-1, -2)
    return A_norm

# ------------- ODE over variable tokens -------------------
class _VarRHS(nn.Module):
    """
    dx/dt = α * (L x) + (A x) Θ     with A,L per-batch (cached)
    x: [B,N,D]
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.theta = nn.Linear(d_model, d_model, bias=False)
        self.alpha_raw = nn.Parameter(torch.tensor(0.1))
        # will be set at forward-call time:
        self.A = None  # [B,N,N]
        self.L = None  # [B,N,N]

    def set_graph(self, A_norm: torch.Tensor):
        B, N, _ = A_norm.shape
        I = torch.eye(N, device=A_norm.device).unsqueeze(0).expand(B, -1, -1)
        self.A = A_norm
        self.L = I - A_norm

    def forward(self, t, x):
        # x: [B,N,D]
        B, N, D = x.shape
        alpha = F.softplus(self.alpha_raw).clamp(max=2.0)
        diff  = alpha * torch.bmm(self.L, x)     # [B,N,D]
        react = self.theta(torch.bmm(self.A, x)) # [B,N,D]
        return diff + react

class _VarPDE(nn.Module):
    def __init__(self, d_model: int, k: int, t_span=(0.0, 0.2), solver="implicit_adams", rtol=1e-4, atol=1e-6):
        super().__init__()
        self.k = k
        self.rhs = _VarRHS(d_model)
        self.register_buffer("tspan", torch.tensor(list(t_span), dtype=torch.float32))
        self.solver = solver; self.rtol = rtol; self.atol = atol

    def forward(self, tokens: torch.Tensor, x_bn: torch.Tensor):
        """
        tokens: [B,N,D]  (GCN+Transformer output)
        x_bn:   [B,N,L]  (raw)
        """
        with torch.no_grad():
            corr = pearson_corr(x_bn)             # [B,N,N]
            A = build_topk_A(corr, self.k)        # [B,N,N]
        self.rhs.set_graph(A)

        t = self.tspan.to(tokens.device)
        y = odeint(self.rhs, tokens, t, method=self.solver)[-1]
        return F.relu(y)

# ------------------ GCN block (yours) ---------------------
class _GCNBlock(nn.Module):
    def __init__(self, d_model, dropout, n_heads, d_ff, num_layers, activation):
        super().__init__()
        self.conv1 = GCNConv(d_model, d_model)
        self.conv2 = GCNConv(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.transformer = TransformerEncoder(d_model, n_heads, num_layers, d_ff, dropout)
        self.act = nn.Sigmoid() if activation == 'sigmoid' else nn.ReLU()

    def forward(self, x_nodes: torch.Tensor, edge_index, B, N, D):
        # x_nodes: [B*N, D] (because of PyG Batch)
        h = self.act(self.conv1(x_nodes, edge_index))
        h = self.dropout(h)
        h = self.act(self.conv2(h, edge_index))
        h = self.dropout(h)
        h = h.reshape(B, N, D)
        # cross-attend: query = original tokens, KV = GCN features
        #h = self.transformer(x_nodes=h, context=h, mask=None)  # simple SA; your encoder supports (q, kv) too
        h = self.transformer(h, h, None)
        return h  # [B,N,D]

# ---------------- main module -----------------------------
class MultiLayerGCN_variate(nn.Module):
    """
    Input:
      enc_out_vari: tokens [B,N,D]
      x_enc:        raw series [B,L,N]
    Output:
      [B,N,D]
    """
    def __init__(self, num_layers, d_model, dropout, n_heads, d_ff, k, activation):
        super().__init__()
        self.d_model = d_model
        self.k = k
        self.layers = nn.ModuleList([
            _GCNBlock(d_model, dropout, n_heads, d_ff, num_layers=1, activation=activation)
        ])
        # PDE refiner across variables
        self.pde = _VarPDE(d_model, k=self.k)

    @torch.no_grad()
    def _edge_index(self, x_bn: torch.Tensor):
        """
        Build a single global edge_index from mean corr over batch (faster).
        x_bn: [B,N,L]
        """
        corr = pearson_corr(x_bn).mean(dim=0)  # [N,N]
        N = corr.size(0)
        idx = torch.topk(corr, k=min(self.k+1, N), dim=-1).indices
        rows = torch.arange(N, device=x_bn.device).unsqueeze(-1).repeat(1, idx.size(1)).reshape(-1)
        cols = idx.reshape(-1)
        ei = torch.stack([rows, cols], dim=0).long()  # [2, E]
        return ei

    def forward(self, enc_out_vari: torch.Tensor, x_enc: torch.Tensor):
        """
        enc_out_vari: [B,N,D]
        x_enc:        [B,L,N]
        """
        B, N, D = enc_out_vari.shape
        x_bn = x_enc.transpose(1, 2).contiguous()  # [B,N,L]
        edge_index = self._edge_index(x_bn)        # [2, E]

        # pack for PyG
        data_list = [Data(x=enc_out_vari[i], edge_index=edge_index) for i in range(B)]
        batch = Batch.from_data_list(data_list)
        h = enc_out_vari

        for layer in self.layers:
            h = layer(batch.x, batch.edge_index, B, N, D)

        # ---- PDE refinement (continuous-time over same graph) ----
        h = self.pde(h, x_bn)  # [B,N,D]
        return h
