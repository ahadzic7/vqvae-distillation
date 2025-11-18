import torch
from torch.distributions import OneHotCategorical, Categorical
import torch.nn.functional as F
from src.vqvae.VQVAE import VQVAE
from src.architectures.arch_registry import get_architecture

class VQVAE_cat(VQVAE):
    def __init__(self, config):
        data = config["input_data"]
        model = config["vqvae"]
        input_shape = data["input_shape"]
        latent_shape = model["latent_shape"]
        arch_fn, e_params, d_params = get_architecture(config)
        super().__init__(
            arch_fn,
            (e_params, d_params),
            (input_shape["channels"], input_shape["height"], input_shape["width"]),
            input_shape["channels"],
            (latent_shape["channels"], latent_shape["height"], latent_shape["width"]),
            model["codebook_size"],
            model["dims"], 
            model["beta"],
        )

    def loss_recons(
            self, 
            batch:torch.Tensor, 
            out:torch.Tensor,
        ):
        dist = OneHotCategorical(probs=out.permute(0,2,3,1))        
        loss_recons = -dist.log_prob(batch.permute(0,2,3,1)).sum(dim=[1,2])
        return loss_recons

    def sample(self, latent):
        probs = torch.empty((latent.shape[0], 256, 28, 28), device=self.device)
        for i, l in enumerate(latent):
            q_e_x = self.codebook.embedding.weight[l].unsqueeze(0).permute(0, 3, 1, 2)
            probs[i] = self.decode(q_e_x)
        return probs

    def params(self, latent):
        return { "probs":self.sample(latent), }
    

    def reconstruction(self, x):
        mode = self.forward(x).argmax(dim=1).unsqueeze(1) / 255.0
        x = x.argmax(dim=1).unsqueeze(1) / 255.0
        return torch.cat([x, mode], 0)
    
    def uniform_params(self, n_components):
        latent = torch.randint(0, self.codebook_size, (n_components,)).to(self.device)
        return self.params(latent)
    
    def categorical_params(self, train_loader, n_components):
        cw_hist = super().codeword_utilization_frequency(train_loader)
        prior = Categorical(probs=F.softmax(cw_hist))
        latent = prior.sample((n_components,)).to(self.device)
        return self.params(latent)
    
    def data_params(self, data, n_components):
        indices = torch.randint(0, len(data), (n_components,))
        inputs = torch.stack([data[i] for i in indices]).to(self.device)
        return { "probs": self.forward(inputs), }
    