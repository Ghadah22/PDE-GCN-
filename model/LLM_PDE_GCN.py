

import torch
import torch.nn as nn

from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
from model.RevIN import RevIN
from layers.GNN_variate import MultiLayerGCN_variate

try:
    from layers.GNN_time import MultiLayerGCN_time
except Exception:
    class MultiLayerGCN_time(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
            self.identity = nn.Identity()
        def forward(self, tokens, x_bn, *args, **kwargs):
            return self.identity(tokens)

class Model(nn.Module):
    """
    Two-branch hybrid:
      - Variate GCN (+Transformer) -> z_var  [B,N,D]
      - Time/patch GCN (+Transformer) -> z_time [B,N,D]
      - Softmax gating + residual fuse
    """
    def __init__(self, configs):
        super().__init__()
        self.seq_len   = configs.seq_len
        self.pred_len  = configs.pred_len
        self.enc_in    = configs.enc_in
        self.c_out     = getattr(configs, "c_out", configs.enc_in)

        self.d_model   = configs.d_model
        self.n_heads   = getattr(configs, "n_heads", 1)
        self.e_layers  = getattr(configs, "e_layers", 1)
        self.d_ff      = getattr(configs, "d_ff", 128)
        self.dropout   = getattr(configs, "dropout", 0.1)
        self.k         = getattr(configs, "k", 3)
        self.activation= getattr(configs, "activation", "gelu")
        self.use_norm  = bool(getattr(configs, "use_norm", 1))

        self.patch_len = int(getattr(configs, "patch_len", 48))
        assert self.seq_len % self.patch_len == 0, "seq_len must be divisible by patch_len"
        self.patch_num = self.seq_len // self.patch_len

        self.revin_layer = RevIN(num_features=self.enc_in, affine=True, subtract_last=False) \
            if self.use_norm else None

        self.value_embedding = DataEmbedding_inverted(
            c_in=self.seq_len, d_model=self.d_model, dropout=self.dropout
        )
        self.pos_emb = PositionalEmbedding(self.d_model)
        self.time_patch_proj = nn.Linear(self.patch_len, self.d_model)

        self.variate_gnn = MultiLayerGCN_variate(
            num_layers=self.e_layers, d_model=self.d_model, dropout=self.dropout,
            n_heads=self.n_heads, d_ff=self.d_ff, k=self.k, activation=self.activation
        )
        self.time_gnn = MultiLayerGCN_time(
            num_layers=self.e_layers, d_model=self.d_model, dropout=self.dropout,
            n_heads=self.n_heads, d_ff=self.d_ff, k=self.k, activation=self.activation
        )

        self.branch_gater = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, 2)
        )
        self.fuse = nn.Sequential(
            nn.Linear(self.d_model * 2, self.d_model),
            nn.GELU(),
            nn.Linear(self.d_model, self.d_model)
        )

        self.to_horizon = nn.Linear(self.d_model, self.pred_len)
        self.var_map = nn.Linear(self.enc_in, self.c_out) if self.c_out != self.enc_in else None

        # a small dict to expose interpretable internals for explanations
        self.last_explain = {}

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if getattr(m, "bias", None) is not None:
                    nn.init.zeros_(m.bias)

    def forecast(self, x_enc, x_mark=None):
        B, L, N = x_enc.shape
        assert L == self.seq_len and N == self.enc_in, f"got {x_enc.shape}, expected [B,{self.seq_len},{self.enc_in}]"

        x_norm = self.revin_layer(x_enc, 'norm') if self.use_norm else x_enc
        tokens = self.value_embedding(x_norm, x_mark)                       # [B,N,D]

        x_bn = x_norm.transpose(1, 2).contiguous()                          # [B,N,L]
        patches = x_bn.unfold(2, self.patch_len, self.patch_len)            # [B,N,P,S]
        time_patch_tokens = self.time_patch_proj(patches)                   # [B,N,P,D]

        z_var  = self.variate_gnn(tokens, x_norm)                           # [B,N,D]
        z_time = self.time_gnn(tokens, x_bn, time_patch_tokens)             # [B,N,D]

        z_cat  = torch.cat([z_var, z_time], dim=-1)                         # [B,N,2D]
        w_log  = self.branch_gater(z_cat)                                   # [B,N,2]
        w      = torch.softmax(w_log, dim=-1)                               # [B,N,2]

        # store interpretable signals
        self.last_explain["branch_weights_mean"] = w.mean(dim=(0,1)).detach().cpu().tolist()  # [2]

        fused = w[...,0:1]*z_var + w[...,1:2]*z_time
        fused = fused + 0.1*self.fuse(z_cat) + 0.2*tokens                   # [B,N,D]

        y = self.to_horizon(fused).transpose(1, 2)                          # [B,pred_len,N]
        if self.var_map is not None:
            y = self.var_map(y)
        if self.use_norm:
            y = self.revin_layer(y, 'denorm')
        return y

    def forward(self, x_enc, x_mark=None):
        return self.forecast(x_enc, x_mark)
