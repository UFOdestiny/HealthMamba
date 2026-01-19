import torch
import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel
from mamba_ssm import Mamba


class AdaptiveGraphGate(nn.Module):

    def __init__(self, d_model, init_bias=-2.0):
        super().__init__()
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
        )
                                                                                           
        nn.init.constant_(self.gate_net[-1].bias, init_bias)

    def forward(self, x):
                                                
        x_mean = x.mean(dim=1, keepdim=True)                
        gate = torch.sigmoid(self.gate_net(x_mean))                
        return gate.expand(-1, x.size(1), -1, -1)                


class LightGraphConv(nn.Module):

    def __init__(self, node_num, d_model, embed_dim=16, base_adj=None, dropout=0.1):
        super().__init__()
        self.node_num = node_num
        self.d_model = d_model

                                                      
        self.node_embed = nn.Parameter(torch.randn(node_num, embed_dim) * 0.02)

                            
        self.msg_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        if base_adj is not None:
            adj = base_adj.clone().detach().float()
                                         
            adj = adj + torch.eye(node_num, device=adj.device, dtype=adj.dtype)
            deg = adj.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            adj = adj / deg
            self.register_buffer("base_adj", adj)
        else:
            self.base_adj = None

    def forward(self, x):
        B, T, N, D = x.shape

                                             
        adaptive_adj = torch.mm(self.node_embed, self.node_embed.t())
        adaptive_adj = F.softmax(adaptive_adj, dim=-1)

        if self.base_adj is not None:
            adj = 0.7 * self.base_adj + 0.3 * adaptive_adj
        else:
            adj = adaptive_adj

                         
        x_flat = x.reshape(B * T, N, D)               
        msg = self.msg_proj(x_flat)               
        agg = torch.matmul(adj, msg)               
        agg = agg.reshape(B, T, N, D)

        return self.norm(self.dropout(agg) + x)


class MambaBlock(nn.Module):

    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return x + self.dropout(self.mamba(self.norm(x)))


class MambaWithFFN(nn.Module):

    def __init__(self, d_model, dropout=0.1, ffn_expand=2):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model)

        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_expand),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ffn_expand, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.mamba(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class GraphGatedMambaLayer(nn.Module):

    def __init__(self, d_model, node_num, embed_dim=16, base_adj=None, dropout=0.1, gate_init=-2.0):
        super().__init__()

                                 
        self.graph_gate = AdaptiveGraphGate(d_model, init_bias=gate_init)
        self.graph_conv = LightGraphConv(node_num, d_model, embed_dim, base_adj, dropout)

                      
        self.mamba = MambaWithFFN(d_model, dropout)

    def forward(self, x):
        B, T, N, D = x.shape

                                    
        gate = self.graph_gate(x)                

                                                                 
        if gate.mean() > 0.01:
            x_graph = self.graph_conv(x)
            x = x + gate * (x_graph - x)

                                                  
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, D)               
        x_flat = self.mamba(x_flat)
        x = x_flat.reshape(B, N, T, D).permute(0, 2, 1, 3)                

        return x


class DeepMambaStack(nn.Module):

    def __init__(self, d_model, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaWithFFN(d_model, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class UNetMamba(BaseModel):

    def __init__(
        self,
        d_model,
        num_layers,
        feature,
        adj=None,
        graph_embed_dim=16,
        dropout=0.1,
        num_graph_layers=1,
        gate_init=-2.0,
        **args
    ):
        super(UNetMamba, self).__init__(**args)
        self.d_model = d_model
        self.num_layers = num_layers
        self.feature = feature

                          
        self.input_proj = nn.Linear(self.feature, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.input_dropout = nn.Dropout(dropout)

                                                                         
        self.graph_mamba_layers = nn.ModuleList([
            GraphGatedMambaLayer(
                d_model=d_model,
                node_num=self.node_num,
                embed_dim=graph_embed_dim,
                base_adj=adj,
                dropout=dropout,
                gate_init=gate_init - 0.5 * i                                         
            )
            for i in range(num_graph_layers)
        ])

                               
        pure_mamba_layers = max(1, num_layers - num_graph_layers)
        self.deep_mamba = DeepMambaStack(d_model, pure_mamba_layers, dropout)

                           
        self.output_norm = nn.LayerNorm(d_model)
        self.time_proj = nn.Linear(self.seq_len, self.horizon)
        self.output_proj = nn.Linear(self.d_model, self.feature)

    def forward(self, x):                
        B, T, N, F = x.shape

                          
        x = self.input_proj(x)                      
        x = self.input_norm(x)
        x = self.input_dropout(x)

                                 
        x_input = x

                                 
        for layer in self.graph_mamba_layers:
            x = layer(x)

                                               
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, self.d_model)               

                         
        x = self.deep_mamba(x)

                      
        x = x.reshape(B, N, T, self.d_model).permute(0, 2, 1, 3)                

                         
        x = x + x_input

                            
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, self.d_model)               
        x = self.output_norm(x)
        x = x.permute(0, 2, 1)               
        x = self.time_proj(x)               
        x = x.permute(0, 2, 1)               
        x = self.output_proj(x)               

                                    
        x = x.reshape(B, N, self.horizon, F)
        x = x.permute(0, 2, 1, 3)

        return x
