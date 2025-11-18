import torch.nn as nn
import torch.nn.functional as F

class CausalConv2d(nn.Module):
    """Causal convolution with mask type A or B"""
    def __init__(self, in_channels, out_channels, kernel_size, mask_type='B'):
        super().__init__()
        self.mask_type = mask_type
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.register_buffer('mask', self.conv.weight.data.clone())
        
        _, _, kH, kW = self.conv.weight.size()
        self.mask.fill_(1)
        self.mask[:, :, kH // 2, kW // 2 + (mask_type == 'B'):] = 0
        self.mask[:, :, kH // 2 + 1:] = 0
        
    def forward(self, x):
        return F.conv2d(
            x, 
            self.conv.weight * self.mask, 
            self.conv.bias,
            stride=self.conv.stride,
            padding=self.conv.padding,
            dilation=self.conv.dilation,
            groups=self.conv.groups
        )
