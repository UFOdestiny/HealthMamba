import torch
import torch.nn as nn
import torch.nn.functional as F
from model.stce import STCE
from model.gmamba import GraphMambaUNet


class HealthMamba(nn.Module):
    def __init__(self, node_num, seq_len, horizon, visit_dim, output_dim,
                 d_hid, d_model, n_scales, n_layers_per_scale, adj,
                 static_dim=16, dynamic_dim=0, embed_dim=16, lam=0.5, dropout=0.1):
        super().__init__()
        self.node_num = node_num
        self.seq_len = seq_len
        self.horizon = horizon
        self.output_dim = output_dim
        self.register_buffer("adj", adj)
        self.stce = STCE(visit_dim, node_num, d_hid, d_model, static_dim, dynamic_dim, dropout)
        self.backbone = GraphMambaUNet(d_model, n_scales, n_layers_per_scale, embed_dim, lam, dropout)
        self.time_proj = nn.Linear(seq_len, horizon)
        self.mu_head = nn.Linear(d_model, output_dim)
        self.log_var_head = nn.Linear(d_model, output_dim)
        self.lower_raw = nn.Linear(d_model, output_dim)
        self.upper_raw = nn.Linear(d_model, output_dim)

    def forward(self, x):
        R = self.stce(x, self.adj)
        Z = self.backbone(R, self.adj)
        Z = Z.permute(0, 2, 3, 1)
        Z = self.time_proj(Z)
        Z = Z.permute(0, 3, 1, 2)
        mu = self.mu_head(Z)
        log_var = self.log_var_head(Z)
        lower = mu - F.softplus(self.lower_raw(Z))
        upper = mu + F.softplus(self.upper_raw(Z))
        return mu, log_var, lower, upper

    def param_num(self):
        return sum(p.numel() for p in self.parameters())
