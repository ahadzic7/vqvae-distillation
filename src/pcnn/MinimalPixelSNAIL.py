import torch
import torch.nn as nn
import torch.nn.functional as F
from src.pcnn.PixelCNNBase import PixelCNNBase
from src.architectures.layers.CausalConv2d import CausalConv2d
from src.architectures.layers.CausalResBlock import CausalResBlock

class MinimalPixelSNAIL(PixelCNNBase):
    def __init__(self, vqvae, config):
        """
        Minimal PixelSNAIL for training on VQ-VAE latents.
        Keeps the structure compatible with your current implementation.
        """
        lat_dim = config.get("hidden_dim", 128)
        n_layers = config.get("n_layers", 2)
        dropout = config.get("dropout", 0.0)

        # Embedding for discrete latents
        embedding = nn.Embedding(vqvae.codebook_size, lat_dim)

        # Layers: initial causal conv + residual blocks
        layers = [CausalConv2d(lat_dim, lat_dim, kernel_size=7, mask_type='A')]
        for _ in range(n_layers):
            layers.append(CausalResBlock(lat_dim, dropout=dropout))

        # Output conv: single 1x1 conv
        output_conv = nn.Conv2d(lat_dim, vqvae.codebook_size, kernel_size=1)

        super().__init__(vqvae, embedding, nn.ModuleList(layers), output_conv)

    def forward(self, x):
        shp = x.size() + (-1,)
        x = self.embedding(x.view(-1)).view(shp)  # (B,H,W,C)
        x = x.permute(0, 3, 1, 2)  # (B,C,H,W)

        for layer in self.layers:
            x = layer(x)

        logits = self.output_conv(x)  # (B,K,H,W)
        return logits

    def loss(self, batch_x):
        with torch.no_grad():
            z_e_x = self.vqvae.encode(batch_x).contiguous()
            latents = self.vqvae.codebook(z_e_x).detach()
        K = self.vqvae.codebook_size
        logits = self.forward(latents).permute(0, 2, 3, 1).contiguous()
        return F.cross_entropy(logits.view(-1, K), latents.view(-1))

    def training_step(self, batch, batch_id):
        loss = self.loss(batch)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_id):
        loss = self.loss(batch)
        self.log('valid_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss