import torch
from src.vae.VAE import VAE
import torch
import torch.nn as nn
from src.architectures.arch_registry import get_architecture
import math


class VAE_con(VAE, nn.Module):
    def __init__(self, config):
        model = config["vae"]
        input_shp = config["input_data"]["input_shape"]
        latent_shp = model["latent_shape"]
        arch_fn, e_params, d_params = get_architecture(config)
        
        odim = input_shp["channels"] * 2
        mini = model["learn_std"]["min_std"]
        maxi = model["learn_std"]["max_std"]
        fs = model["filters"]
        fs2 = [f // 2 for f in fs]

        e, d = arch_fn(
            (e_params, d_params), 
            input_shp["channels"], 
            odim, 
            model["latent_shape"]["channels"], 
            fs,
            fs2,
            vae=True,
            mu_act=model["mu_activation"],
            min_max_std=(mini, maxi)
        )
        
        super().__init__(e, d, input_shp, odim, latent_shp, model["beta"])
        self.mu_act=model["mu_activation"],
        self.min_max_std=model["learn_std"]

        self.L2_PI = torch.tensor(math.log(2*math.pi)).to("cuda")

    def loss_recons(self, bx, out):
        mu, log_var = out.chunk(2, dim=1)
        log_sd = 0.5 * log_var
        ll = -0.5 * (((bx - mu) / log_sd.exp()).pow(2) + log_var + self.L2_PI)
        return -torch.sum(ll, dim=[1, 2, 3])
 


    @torch.no_grad()
    def params(self, latent=None, n_components=None):
        if latent is None and n_components is not None:
            latent, _ = self._params(n_components)
        elif latent is None and n_components is None:
            raise Exception("LATENT VS SAMPLING LATENT!")

        mus, lvs = [], []
        chunk_size = 2**10
        for bl in torch.split(latent, chunk_size):
            mc, lvc = self.decode(bl).chunk(2, dim=1)
            mus.append(mc)
            lvs.append(lvc)

        return {
            "mean": torch.cat(mus, dim=0),
            "sd": torch.cat(lvs, dim=0).mul(0.5).exp()
        }
    

    def reconstruction(self, x):
        mu, _ = self.forward(x).chunk(2, dim=1)
        return torch.cat([x, mu], 0).cpu()

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

