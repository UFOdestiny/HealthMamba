import torch
import torch.nn as nn
import torch.nn.functional as F
from base.model import BaseModel
from mamba_ssm import Mamba


class TemporalConvBlock(nn.Module):
    def __init__(self, d_model, kernel_size=3, dropout=0.1):
        super().__init__()
        self.conv = nn.Conv1d(
            d_model, d_model,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=1
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = x.permute(0, 2, 1)             
        x = self.conv(x)
        x = x.permute(0, 2, 1)             
        x = self.dropout(x)
        return self.norm(x + residual)


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


class MultiScaleMambaBlock(nn.Module):

    def __init__(self, d_model, num_scales=3, dropout=0.1):
        super().__init__()
        self.num_scales = num_scales

                                  
        self.scale_mambas = nn.ModuleList([
            Mamba(d_model=d_model) for _ in range(num_scales)
        ])

                      
        self.fusion = nn.Linear(d_model * num_scales, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, D = x.shape
        residual = x

        scale_outputs = []
        for i, mamba in enumerate(self.scale_mambas):
            scale = 2 ** i              

            if scale == 1:
                                
                out = mamba(x)
            else:
                                                 
                if T >= scale:
                                            
                    x_down = x[:, ::scale, :]                    
                    out_down = mamba(x_down)
                                                      
                    out = F.interpolate(
                        out_down.permute(0, 2, 1),
                        size=T,
                        mode='linear',
                        align_corners=False
                    ).permute(0, 2, 1)
                else:
                    out = mamba(x)

            scale_outputs.append(out)

                                 
        x = torch.cat(scale_outputs, dim=-1)                        
        x = self.fusion(x)             
        x = self.dropout(x)

        return self.norm(x + residual)


class HierarchicalMambaStage(nn.Module):

    def __init__(self, d_model, num_blocks, dropout=0.1, ffn_expand=2, use_temporal_conv=True):
        super().__init__()
        self.use_temporal_conv = use_temporal_conv

        if use_temporal_conv:
            self.temporal_conv = TemporalConvBlock(d_model, kernel_size=3, dropout=dropout)

        self.blocks = nn.ModuleList([
            MambaWithFFN(d_model, dropout, ffn_expand)
            for _ in range(num_blocks)
        ])

    def forward(self, x):
        if self.use_temporal_conv:
            x = self.temporal_conv(x)

        for block in self.blocks:
            x = block(x)

        return x


class UNetMamba(BaseModel):

    def __init__(
        self,
        d_model,
        num_layers,
        sample_factor,                                            
        feature,
        dropout=0.1,
        ffn_expand=2,
        use_multiscale=True,
        use_temporal_conv=True,
        **args
    ):
        super(UNetMamba, self).__init__(**args)
        self.d_model = d_model
        self.num_layers = num_layers
        self.feature = feature
        self.use_multiscale = use_multiscale

                          
        self.input_proj = nn.Linear(self.feature, self.d_model)
        self.input_norm = nn.LayerNorm(self.d_model)
        self.input_dropout = nn.Dropout(dropout)

                                   
        blocks_per_stage = max(1, num_layers // 2)                              
        num_stages = max(1, (num_layers + blocks_per_stage - 1) // blocks_per_stage)

        self.stages = nn.ModuleList([
            HierarchicalMambaStage(
                d_model=d_model,
                num_blocks=blocks_per_stage,
                dropout=dropout,
                ffn_expand=ffn_expand,
                use_temporal_conv=use_temporal_conv and (i == 0)                                                    
            )
            for i in range(num_stages)
        ])

                                      
        if use_multiscale:
            self.multiscale_block = MultiScaleMambaBlock(
                d_model=d_model,
                num_scales=min(sample_factor, 3),                
                dropout=dropout
            )

                           
        self.final_mamba = Mamba(d_model=d_model)
        self.final_norm = nn.LayerNorm(d_model)

                           
        self.output_norm = nn.LayerNorm(d_model)
        self.time_proj = nn.Linear(self.seq_len, self.horizon)
        self.output_proj = nn.Linear(self.d_model, self.feature)

    def forward(self, x):                
        B, T, N, F = x.shape

                              
        x = x.permute(0, 2, 1, 3).reshape(B * N, T, F)

                          
        x = self.input_proj(x)
        x = self.input_norm(x)
        x = self.input_dropout(x)

                                              
        x_input = x

                                   
        for stage in self.stages:
            x = stage(x)

                                         
        if self.use_multiscale:
            x = self.multiscale_block(x)

                               
        x = x + x_input

                                
        x = self.final_mamba(self.final_norm(x)) + x

                           
        x = self.output_norm(x)
        x = x.permute(0, 2, 1)               
        x = self.time_proj(x)               
        x = x.permute(0, 2, 1)               
        x = self.output_proj(x)               

                                    
        x = x.reshape(B, N, self.horizon, F)
        x = x.permute(0, 2, 1, 3)

        return x
