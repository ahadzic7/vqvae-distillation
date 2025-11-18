import torch.nn as nn
from src.architectures.layers.CausalConv2d import CausalConv2d

class CausalResBlock(nn.Module):
    """Residual block with causal convolutions"""
    def __init__(self, channels, kernel_size=3, dropout=0):
        super().__init__()
        self.layer_norm = nn.LayerNorm([channels])
        self.block = nn.Sequential(
            CausalConv2d(channels, channels, kernel_size),
            nn.GELU(),
            nn.Dropout(dropout),
            CausalConv2d(channels, channels, kernel_size),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        x_tmp = self.layer_norm(x.permute(0, 2, 3, 1)) # LayerNorm expects (B, H, W, C)
        return x + self.block(x_tmp.permute(0, 3, 1, 2))