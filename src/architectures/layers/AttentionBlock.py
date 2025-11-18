import torch.nn as nn
from src.architectures.layers.CausalAttention import CausalAttention

class AttentionBlock(nn.Module):
    """Attention block with residual connection"""
    def __init__(self, channels, n_heads, dropout=0):
        super().__init__()
        self.norm = nn.LayerNorm([channels])
        self.attn = CausalAttention(channels, n_heads)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x_tmp = self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) 
        x_tmp = self.attn(x_tmp)
        x_tmp = self.dropout(x_tmp)
        return x_tmp + x