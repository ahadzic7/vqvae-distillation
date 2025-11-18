import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalAttention(nn.Module):
    """Causal self-attention with proper 2D raster-scan masking"""
    def __init__(self, in_channels, n_heads=4, head_dim=None, H=8, W=8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim or in_channels // n_heads
        self.scale = self.head_dim ** -0.5
        self.H, self.W = H, W

        self.to_qkv = nn.Conv2d(in_channels, 3 * n_heads * self.head_dim, 1)
        self.to_out = nn.Conv2d(n_heads * self.head_dim, in_channels, 1)

        # Precompute the 2D causal mask and register as buffer
        mask = torch.tril(torch.ones(1, 1, H * W, H * W))
        # mask = self.build_causal_mask(height, width)
        self.register_buffer("mask", mask, persistent=False)

    @staticmethod
    def build_causal_mask(H, W):
        mask = torch.zeros(H * W, H * W)
        for i in range(H):
            for j in range(W):
                idx = i * W + j
                # allow all pixels in previous rows
                mask[idx, :i*W] = 1
                # allow all previous pixels in the same row + itself
                mask[idx, i*W : i*W + j + 1] = 1
        return mask.unsqueeze(0).unsqueeze(0)  # (1,1,N,N)

    def forward(self, x):
        B, _, H, W = x.shape
        assert H == self.H and W == self.W, "Input shape must match mask size"

        qkv = self.to_qkv(x)
        q, k, v = qkv.chunk(3, dim=1)

        # Flatten to sequence (row-major order)
        N = H * W
        q = q.view(B, self.n_heads, self.head_dim, N).transpose(2, 3)  # (B, nH, N, d)
        k = k.view(B, self.n_heads, self.head_dim, N).transpose(2, 3)  # (B, nH, N, d)
        v = v.view(B, self.n_heads, self.head_dim, N).transpose(2, 3)  # (B, nH, N, d)

        # attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B,nH,N,N)
        # attn = attn.masked_fill(self.mask[:, :, :N, :N] == 0, float("-inf"))
        # attn = F.softmax(attn, dim=-1)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        mask = self.mask[:, :, :N, :N] == 0
        attn = attn.masked_fill(mask, float("-inf"))   # safer than -inf
        attn = F.softmax(attn, dim=-1)

        out = torch.matmul(attn, v).transpose(2, 3).contiguous() # (B,nH,N,d)
        out = out.view(B, self.n_heads * self.head_dim, H, W)

        return self.to_out(out)
