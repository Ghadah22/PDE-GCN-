import torch
import torch.nn as nn

from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
from model.RevIN import RevIN
from layers.GNN_variate import MultiLayerGCN_variate

try:
    from layers.GNN_time import MultiLayerGCN_time
except Exception:
    class MultiLayerGCN_time(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__(); self.identity = nn.Identity()
        def forward(self, tokens, x_bn, *args, **kwargs): return self.identity(tokens)


class Model(nn.Module):
    """
    DFGCN hybrid with hooks for LLM explanations & covariate fusion.
    """
    def __init__(self, configs):
        super().__init__()
        # shapes
        self.seq_len   = configs.seq_len
        self.pred_len  = configs.pred_len
        self.enc_in    = configs.enc_in
        self.c_out     = getattr(configs, "c_out", configs.enc_in)

        # hparams
        self.d_model   = configs.d_model
        self.n_heads   = getattr(configs, "n_heads", 1)
        self.e_layers  = getattr(configs, "e_layers", 1)
        self.d_ff      = getattr(configs, "d_ff", 128)
        self.dropout   = getattr(configs, "dropout", 0.1)
        self.k         = getattr(configs, "k", 3)
        self.activation= getattr(configs, "activation", "gelu")
        self.use_norm  = bool(getattr(configs, "use_norm", 1))

        # ### NEW: small flags so you can control outputs from args
        self.return_aux    = bool(getattr(configs, "return_aux", False))  # return (y, aux) if True
        self.use_global_pde = False  # keep False unless you re-enable your top ODE block

        # patching for time branch
        self.patch_len = int(getattr(configs, "patch_len", 48))
        assert self.seq_len % self.patch_len == 0, \
            f"seq_len {self.seq_len} must be divisible by patch_len {self.patch_len}"
        self.patch_num = self.seq_len // self.patch_len

        # RevIN
        self.revin_layer = RevIN(num_features=self.enc_in, affine=True, subtract_last=False) \
            if self.use_norm else None

        # embeddings
        self.value_embedding = DataEmbedding_inverted(
            c_in=self.seq_len, d_model=self.d_model, dropout=self.dropout
        )
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.time_patch_proj = nn.Linear(self.patch_len, self.d_model)

        # branches
        self.variate_gnn = MultiLayerGCN_variate(
            num_layers=self.e_layers, d_model=self.d_model, dropout=self.dropout,
            n_heads=self.n_heads, d_ff=self.d_ff, k=self.k, activation=self.activation
        )
        self.time_gnn = MultiLayerGCN_time(
            num_layers=self.e_layers, d_model=self.d_model, dropout=self.dropout,
            n_heads=self.n_heads, d_ff=self.d_ff, k=self.k, activation=self.activation
        )

        # gating + fuse
        in_gater = self.d_model * (3 if self.use_global_pde else 2)
        out_heads = (3 if self.use_global_pde else 2)
        self.branch_gater = nn.Sequential(
            nn.Linear(in_gater, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, out_heads)
        )
        self.fuse = nn.Sequential(
            nn.Linear(in_gater, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model)
        )

        # ### NEW: optional covariate fusion for future offline LLM features
        # If you later provide horizon-aligned numeric features (H, C), set configs.cov_dim>0
        self.cov_dim = int(getattr(configs, "cov_dim", 0))
        if self.cov_dim > 0:
            # FiLM-style scales for each variable over horizon
            self.cov_proj = nn.Linear(self.cov_dim, self.d_model)
            self.cov_gate = nn.Sequential(nn.Linear(self.d_model, self.d_model), nn.Tanh())

        # head
        self.to_horizon = nn.Linear(self.d_model, self.pred_len)
        self.var_map = nn.Linear(self.enc_in, self.c_out) if self.c_out != self.enc_in else None

        # init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

        # ### NEW: a small cache that the explainer reads
        self.last_explain = {}  # filled each forward()

    def _apply_covariates(self, fused: torch.Tensor, cov_future: torch.Tensor):
        """
        cov_future: [B, H, C] numeric features (e.g. offline LLM flags per future step)
        fused:      [B, N, D]
        Returns:    fused' used before horizon head (broadcast FiLM)
        """
        if self.cov_dim == 0 or cov_future is None:
            return fused
        # Project cov → D, average across H to get one modulation vector per sample.
        # If you prefer step-wise FiLM, move this after to_horizon.
        z = self.cov_proj(cov_future)         # [B, H, D]
        z = z.mean(dim=1, keepdim=False)      # [B, D]
        g = self.cov_gate(z)                  # [B, D] in (-1,1)
        return fused + g.unsqueeze(1)         # broadcast over N

    def forecast(self, x_enc, x_mark=None, cov_future: torch.Tensor = None):
        """
        x_enc: [B, L, N]
        cov_future (optional): [B, pred_len, C] numeric future features
        """
        B, L, N = x_enc.shape
        assert L == self.seq_len and N == self.enc_in, f"got {x_enc.shape}, expected [B,{self.seq_len},{self.enc_in}]"

        # RevIN
        x_norm = self.revin_layer(x_enc, 'norm') if self.use_norm else x_enc

        # per-variable tokens [B,N,D]
        tokens = self.value_embedding(x_norm, x_mark)           # [B,N,D]

        # per-variable, per-patch tokens [B,N,P,D]
        x_bn = x_norm.transpose(1, 2).contiguous()              # [B,N,L]
        patches = x_bn.unfold(2, self.patch_len, self.patch_len)  # [B,N,P,S]
        time_patch_tokens = self.time_patch_proj(patches)       # [B,N,P,D]

        # branches
        z_var  = self.variate_gnn(tokens, x_norm)               # [B,N,D]
        z_time = self.time_gnn(tokens, x_bn, time_patch_tokens) # [B,N,D]

        if self.use_global_pde:
            z_cat  = torch.cat([z_var, z_time], dim=-1)  # if you re-enable the third stream
        else:
            z_cat  = torch.cat([z_var, z_time], dim=-1)         # [B,N,2D]

        w_log  = self.branch_gater(z_cat)                       # [B,N,2] or [B,N,3]
        w      = torch.softmax(w_log, dim=-1)

        # ### NEW: cache rich info for LLM explanations
        # global avg weights and per-var average weights
        self.last_explain = {
            "branch_weights": w.mean(dim=(0,1)).detach().cpu().tolist(),    # [2] or [3]
            "branch_weights_per_var": w.mean(dim=0).detach().cpu().tolist() # [N, 2/3]
        }

        if self.use_global_pde:
            fused = w[...,0:1]*z_var + w[...,1:2]*z_time + w[...,2:3]
        else:
            fused = w[...,0:1]*z_var + w[...,1:2]*z_time

        fused = fused + 0.1*self.fuse(z_cat) + 0.2*tokens      # [B,N,D]

        # ### NEW: optional covariate FiLM (future numeric features)
        fused = self._apply_covariates(fused, cov_future)      # [B,N,D]

        # to horizon
        y = self.to_horizon(fused).transpose(1, 2)             # [B,pred_len,N]
        if self.var_map is not None:
            y = self.var_map(y)                                # [B,pred_len,c_out]
        if self.use_norm:
            y = self.revin_layer(y, 'denorm')

        # ### NEW: optionally return aux for explainer/debug
        if self.return_aux:
            aux = {
                "z_var_sample_mean": z_var.mean().detach().cpu().item(),
                "z_time_sample_mean": z_time.mean().detach().cpu().item(),
                "tokens_std": tokens.std().detach().cpu().item(),
            }
            return y, aux
        return y

    def forward(self, x_enc, x_mark=None, cov_future: torch.Tensor = None):
        return self.forecast(x_enc, x_mark, cov_future)
