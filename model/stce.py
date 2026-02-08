import torch
import torch.nn as nn


class STCE(nn.Module):
    def __init__(self, visit_dim, node_num, d_hid, d_model, static_dim=16, dynamic_dim=0, dropout=0.1):
        super().__init__()
        self.dynamic_dim = dynamic_dim
        self.visit_proj = nn.Sequential(nn.Linear(visit_dim, d_hid), nn.SiLU(), nn.Dropout(dropout))
        self.node_emb = nn.Parameter(torch.randn(node_num, static_dim) * 0.02)
        self.static_proj = nn.Sequential(nn.Linear(static_dim, d_hid), nn.SiLU(), nn.Dropout(dropout))
        if dynamic_dim > 0:
            self.dynamic_proj = nn.Sequential(nn.Linear(dynamic_dim, d_hid), nn.SiLU(), nn.Dropout(dropout))
        self.gconv_w = nn.Linear(d_hid, d_hid)
        self.gconv_b = nn.Parameter(torch.zeros(d_hid))
        self.depth_conv = nn.Conv1d(d_hid, d_hid, kernel_size=3, padding=1, groups=d_hid)
        self.temporal_norm = nn.LayerNorm(d_hid)
        self.channel_mlp = nn.Sequential(nn.Linear(d_hid, d_hid * 2), nn.SiLU(), nn.Linear(d_hid * 2, d_hid))
        self.proj_norm = nn.LayerNorm(d_hid * 2)
        self.out_proj = nn.Sequential(nn.Linear(d_hid * 2, d_model), nn.SiLU(), nn.Dropout(dropout))

    def forward(self, V, adj, E=None):
        B, T, N, _ = V.shape
        Hv = self.visit_proj(V)
        Hd = self.static_proj(self.node_emb)
        X = Hv + Hd
        if self.dynamic_dim > 0 and E is not None:
            X = X + self.dynamic_proj(E)
        Xf = X.reshape(B * T, N, -1)
        adj_exp = adj.unsqueeze(0).expand(B * T, -1, -1)
        Xf = torch.relu(torch.bmm(adj_exp, self.gconv_w(Xf)) + self.gconv_b)
        X_tilde = Xf.reshape(B, T, N, -1)
        d_hid = X_tilde.shape[-1]
        h = X_tilde.permute(0, 2, 3, 1).reshape(B * N, d_hid, T)
        U = self.depth_conv(h).reshape(B, N, d_hid, T).permute(0, 3, 1, 2)
        U = U + X_tilde
        X_hat = self.channel_mlp(self.temporal_norm(U)) + U
        Hd_exp = Hd.unsqueeze(0).unsqueeze(0).expand(B, T, -1, -1)
        Z = torch.cat([X_hat, Hd_exp], dim=-1)
        R = self.out_proj(self.proj_norm(Z))
        return R
