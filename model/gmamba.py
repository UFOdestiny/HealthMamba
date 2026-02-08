import torch
import torch.nn as nn
import torch.nn.functional as F
from mamba_ssm import Mamba


class AdaptiveGraphLearning(nn.Module):
    def __init__(self, d_model, embed_dim=16, lam=0.5):
        super().__init__()
        self.W = nn.Linear(d_model, embed_dim)
        self.a = nn.Linear(embed_dim * 2, 1)
        self.lam = lam

    def forward(self, X, adj_prior=None):
        u = X.mean(dim=1)
        h = self.W(u)
        B, N, E = h.shape
        hi = h.unsqueeze(2).expand(-1, -1, N, -1)
        hj = h.unsqueeze(1).expand(-1, N, -1, -1)
        e = F.leaky_relu(self.a(torch.cat([hi, hj], dim=-1)).squeeze(-1))
        alpha = F.softmax(e, dim=-1)
        A_sym = 0.5 * (alpha + alpha.transpose(-1, -2))
        deg = A_sym.sum(dim=-1).clamp(min=1e-6)
        D_inv_sqrt = deg.pow(-0.5)
        A_hat = D_inv_sqrt.unsqueeze(-1) * A_sym * D_inv_sqrt.unsqueeze(-2)
        if adj_prior is not None:
            A0 = adj_prior.unsqueeze(0).expand(B, -1, -1)
            return self.lam * A0 + (1 - self.lam) * A_hat
        return A_hat


class GMambaBlock(nn.Module):
    def __init__(self, d_model, embed_dim=16, lam=0.5, dropout=0.1):
        super().__init__()
        self.agl = AdaptiveGraphLearning(d_model, embed_dim, lam)
        self.spatial_proj = nn.Linear(d_model, d_model)
        self.spatial_bias = nn.Parameter(torch.zeros(d_model))
        self.ssm = Mamba(d_model=d_model)
        self.channel_norm = nn.LayerNorm(d_model)
        self.channel_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.SiLU(), nn.Linear(d_model * 2, d_model)
        )
        self.out_proj = nn.Linear(d_model, d_model)
        self.out_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, X, adj_prior=None):
        B, T, N, D = X.shape
        A = self.agl(X, adj_prior)
        Xf = X.reshape(B * T, N, D)
        Af = A.unsqueeze(1).expand(-1, T, -1, -1).reshape(B * T, N, N)
        G = torch.relu(torch.bmm(Af, self.spatial_proj(Xf)) + self.spatial_bias)
        G = G.reshape(B, T, N, D) + X
        Gn = G.permute(0, 2, 1, 3).reshape(B * N, T, D)
        Tn = self.ssm(Gn) + Gn
        Tn = Tn.reshape(B, N, T, D).permute(0, 2, 1, 3)
        C = self.channel_mlp(self.channel_norm(Tn)) + Tn
        O = self.out_norm(self.out_proj(C)) + X
        return self.dropout(O)


class GMambaStage(nn.Module):
    def __init__(self, d_model, n_layers, embed_dim=16, lam=0.5, dropout=0.1):
        super().__init__()
        self.blocks = nn.ModuleList([
            GMambaBlock(d_model, embed_dim, lam, dropout) for _ in range(n_layers)
        ])

    def forward(self, X, adj_prior=None):
        for block in self.blocks:
            X = block(X, adj_prior)
        return X


class GraphMambaUNet(nn.Module):
    def __init__(self, d_model, n_scales, n_layers_per_scale, embed_dim=16, lam=0.5, dropout=0.1):
        super().__init__()
        self.n_scales = n_scales
        self.enc_stages = nn.ModuleList([
            GMambaStage(d_model, n_layers_per_scale, embed_dim, lam, dropout)
            for _ in range(n_scales)
        ])
        self.downsamples = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=3, stride=2, padding=1)
            for _ in range(n_scales - 1)
        ])
        self.upsamples = nn.ModuleList([
            nn.ConvTranspose1d(d_model, d_model, kernel_size=4, stride=2, padding=1)
            for _ in range(n_scales - 1)
        ])
        self.fuse_projs = nn.ModuleList([
            nn.Linear(d_model * 2, d_model) for _ in range(n_scales - 1)
        ])
        self.dec_stages = nn.ModuleList([
            GMambaStage(d_model, n_layers_per_scale, embed_dim, lam, dropout)
            for _ in range(n_scales - 1)
        ])

    def _conv_reshape(self, X, layer):
        B, T, N, D = X.shape
        Xr = X.permute(0, 2, 3, 1).reshape(B * N, D, T)
        Xr = layer(Xr)
        T2 = Xr.shape[-1]
        return Xr.reshape(B, N, D, T2).permute(0, 3, 1, 2)

    @staticmethod
    def _match(x, target_len):
        T = x.shape[1]
        if T == target_len:
            return x
        if T > target_len:
            return x[:, :target_len]
        return F.pad(x, (0, 0, 0, 0, 0, target_len - T))

    def forward(self, R, adj_prior=None):
        skips = []
        X = R
        for s in range(self.n_scales):
            X = self.enc_stages[s](X, adj_prior)
            if s < self.n_scales - 1:
                skips.append(X)
                X = self._conv_reshape(X, self.downsamples[s])
        Z = X
        for s in range(self.n_scales - 2, -1, -1):
            Z = self._conv_reshape(Z, self.upsamples[s])
            Y_enc = skips[s]
            Z = self._match(Z, Y_enc.shape[1])
            Z = self.fuse_projs[s](torch.cat([Y_enc, Z], dim=-1))
            Z = self.dec_stages[s](Z, adj_prior)
        return Z
