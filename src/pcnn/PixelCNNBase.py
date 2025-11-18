from abc import ABC, abstractmethod

import torch
import pytorch_lightning as pl

class PixelCNNBase(pl.LightningModule, ABC):
    def __init__(self, vqvae, embedding, layers, output_conv):
        super().__init__()
        self.lat_dim = vqvae.latent_shape[0]
        self.embedding = embedding
        self.layers = layers
        self.output_conv = output_conv
        self.vqvae = vqvae
        self.save_hyperparameters(ignore=['vqvae', 'output_conv', 'layers', 'embedding'])

    @abstractmethod
    def forward(self, x, *args, **kwargs):
        """Implement in subclass: defines autoregressive pass."""
        pass

    @abstractmethod
    def loss(self, *args, **kwargs):
        """Implement in subclass: defines training loss."""
        pass

    def configure_optimizers(self):
        # og
        return torch.optim.Adam(self.parameters(), lr=3e-4, weight_decay=1e-5)
    
        # not too shaby has some potential
        # return torch.optim.AdamW(self.parameters(), lr=3e-4, weight_decay=1e-4)




    def count_parameters(self, component_names=None):
        return sum(
            param.numel() for name, param in self.named_parameters()
            if component_names is None or any(name.startswith(cn) for cn in component_names)
        )
    
    @torch.no_grad()
    def cartesian_product_batch(self, batch_size=2**10):
        A = torch.arange(self.vqvae.codebook_size, device=self.device)
        s = self.vqvae.latent_shape[1]
        d = len(A)
        n = s ** 2
        num_el = d ** n

        for i in range(0, num_el, batch_size):
            batch_indices = torch.arange(i, min(i + batch_size, num_el))
            indices = [(batch_indices // (d ** (n - j - 1))) % d for j in range(n)]
            yield A[torch.stack(indices, dim=1)].reshape(-1, s, s)