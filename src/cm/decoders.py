import torch
from torch.distributions import Normal
from typing import Optional
from src.cm.utils.losses import mse_loss, ce_loss
import torch.nn as nn
import torch.nn.functional as F


class CategoricalDecoder(nn.Module):
    def __init__(self, net: nn.Module):
        super(CategoricalDecoder, self).__init__()
        self.net = net

    def decode_latent(self, z: torch.Tensor, n_chunks: Optional[int]=None):
        z_chunks = tuple([z]) if n_chunks is None else z.chunk(n_chunks, dim=0)
        with torch.no_grad():
            return torch.stack([self.net(z_chunk) for z_chunk in z_chunks])

    def forward(
            self, 
            x: torch.Tensor, 
            log_w: torch.Tensor, 
            z: torch.Tensor, 
            k: Optional[int]=None, 
            missing: Optional[bool]=None, 
            n_chunks: Optional[int]=None,
        ):
        recons = self.decode_latent(z, n_chunks)
        log_prob_bins = torch.cat([ce_loss(r, x, k=None, missing=missing) for r in recons], dim=1)
        log_prob_bins += log_w.unsqueeze(0)
        
        if k is not None:
            z_top_k = z[torch.topk(log_prob_bins, k=k, dim=-1)[1]]  # shape (batch_size, k, latent_dim)
            return ce_loss(self.net(z_top_k.view(x.shape[0] * k, -1)), x, k=k, missing=missing)
        
        return log_prob_bins


class GaussianDecoder(nn.Module):
    def __init__(
            self, 
            net, 
            learn_std: bool = True, 
            min_std: float = 0.1, 
            max_std: float = 1.0, 
            mu_activation=nn.Identity()
        ):
        super(GaussianDecoder, self).__init__()
        self.net = net
        self.learn_std = learn_std
        self.min_std = min_std
        self.max_std = max_std
        self.mu_activation = mu_activation

    def forward(
            self, 
            x: torch.Tensor, 
            log_w: torch.Tensor, 
            z: torch.Tensor, 
            k: Optional[int] = None, 
            missing: Optional[bool] = None, 
            n_chunks: Optional[int] = None
        ):
        z_chunks = tuple([z]) if n_chunks is None else z.chunk(n_chunks, dim=0)
        batch_size = len(x)

        if k is not None:
            with torch.no_grad(): # Run approximate posterior to find the 'best' k z values for each x
                log_prob_bins = torch.cat([mse_loss(*self._get_params(z_chunk), x, k=None, missing=missing) for z_chunk in z_chunks], dim=1)
                log_prob_bins = log_prob_bins + log_w.unsqueeze(0)
                z_top_k = z[torch.topk(log_prob_bins, k=k, dim=-1)[1]]  # shape (batch_size, k, latent_dim)
            log_prob_bins_top_k = mse_loss(*self._get_params(z_top_k.view(batch_size * k, -1)), x, k=k, missing=missing)
            return log_prob_bins_top_k
        
        log_prob_bins = torch.cat([mse_loss(*self._get_params(z_chunk), x, k, missing) for z_chunk in z_chunks], dim=1)
        return log_prob_bins + log_w.unsqueeze(0)

    def _get_params(self, z):
        raw_out = self.net(z)
        if torch.isnan(raw_out).any():
            print("NaN in net output, batch size:", z.shape[0])
            raise Exception("NaN in net output")
        if torch.isinf(raw_out).any():
            print("Inf in net output, batch size:", z.shape[0])
            raise Exception("NaN in net output")
        
        if self.learn_std:
            mu, logvar = raw_out.chunk(2, dim=1)
            mu = self.mu_activation(mu)
            std = F.softplus(logvar) + self.min_std
            std = torch.clamp(std, min=self.min_std, max=self.max_std)
        else:
            mu = self.mu_activation(raw_out)
            std = torch.full_like(mu, fill_value=self.min_std)
        return mu, std

    def sample_continuous(self, z, components, std_correction:float = 1.):
        mu, std = self._get_params(z)
        return Normal(mu[components], std[components] * std_correction).sample()

    def sample_mixture(self, z:torch.Tensor, log_w:torch.Tensor, std_correction:float=1., device= 'cuda',):
        if self.learn_std:
            raise Exception('Not implemented.')
        else:
            components = self.mu_activation(self.net(z.to(device)))
            idx = torch.distributions.Categorical(logits=log_w).sample([z.size(0)])
            samples = components[idx] + torch.randn_like(components[idx]) * self.min_std * std_correction
        return samples




