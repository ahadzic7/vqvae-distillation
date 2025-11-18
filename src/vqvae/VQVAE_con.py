import torch
import torch.nn as nn
import torch.nn.functional as F
from src.vqvae.VQVAE import VQVAE
from src.architectures.arch_registry import get_architecture
import math
from torch.distributions import Normal, Categorical

class VQVAE_con(VQVAE, nn.Module):
    def __init__(self, config):
        model = config["vqvae"]
        input_shp = config["input_data"]["input_shape"]
        latent_shp = model["latent_shape"]
        arch_fn, e_params, d_params = get_architecture(config)
        
        odim = input_shp["channels"] * 2
        mini = model["learn_std"]["min_std"]
        maxi = model["learn_std"]["max_std"]
        e, d = arch_fn(
            (e_params, d_params), 
            input_shp["channels"], 
            odim, 
            model["latent_shape"]["channels"], 
            model["filters"], 
            model["filters"], 
            mu_act=model["mu_activation"],
            min_max_std=(mini, maxi)
        )
        
        K = model["codebook_size"]
        super().__init__(e, d, input_shp, odim, latent_shp, K, model["beta"])
        self.mu_act=model["mu_activation"]
        self.min_max_std=model["learn_std"]

        self.L2_PI = torch.tensor(math.log(2*math.pi)).to(self.device)


    def loss_recons(self, bx, out):
        mu, log_var = out.chunk(2, dim=1)
        ll = -0.5 * (log_var + (bx - mu).pow(2) / log_var.exp() + self.L2_PI)
        return -torch.sum(ll, dim=[1, 2, 3])

 
 
 
    @torch.no_grad()
    def params(self, latent, chunk_size=2**10):
        mus, lvs = [], []
        weights = self.codebook.embedding.weight
        for bl in torch.split(latent, chunk_size):
            q_e_x = weights[bl.long()].permute(0, 3, 1, 2)
            mc, lvc = self.decode(q_e_x).chunk(2, dim=1)
            mus.append(mc)
            lvs.append(lvc)

        return {
            "mean": torch.cat(mus, dim=0),
            "sd": torch.cat(lvs, dim=0).mul(0.5).exp()
        }
    

    def reconstruction(self, x):
        mu, _ = self.forward(x).chunk(2, dim=1)
        return torch.cat([x, mu], 0).cpu()


    def uniform_params(self, n_components):
        latent = torch.randint(0, self.codebook_size, (n_components,)).to(self.device)
        return self.params(latent)

    def data_params(self, data, n_components):
        indices = torch.randint(0, len(data), (n_components,))
        inputs = torch.stack([data[i] for i in indices]).to(self.device)
        mu, log_var = self.forward(inputs).chunk(2, dim=1)
        return {
            "mean": mu,
            "sd": log_var.mul(0.5).exp()
        }
    
    def sample(self, latent):
        return self.params(latent)["mean"]