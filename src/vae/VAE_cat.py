import torch
from torch.distributions import Normal, OneHotCategorical
from src.vae.VAE_arch import vae_cat
from src.vae.VAE import VAE
from tqdm import tqdm

class VAE_cat(VAE):
    def __init__(
            self, 
            architecture,
            input_dim,
            output_dim,
            latent_dim,
            dims, 
            beta
        ):
        super().__init__(
            init_f=architecture,
            input_dim=input_dim,
            output_dim=output_dim,
            latent_dim=latent_dim,
            dims=dims, 
            beta=beta
        )

    def loss(self, batch:torch.Tensor, out_d:torch.Tensor, kl_div):
        batch = batch.permute(0,2,3,1)
        dist = OneHotCategorical(probs=out_d.permute(0, 2, 3, 1))
        rec_loss = -dist.log_prob(batch).sum(dim=[1,2]).mean()
        return rec_loss + self.beta * kl_div

    def reconstruction(self, x):
        mode = self.forward(x).argmax(dim=1).unsqueeze(1) / 255.0
        x = x.argmax(dim=1).unsqueeze(1) / 255.0
        return torch.cat([x, mode], 0).cpu()
    
    def sample(self, n):
        latent = Normal(0, 1).sample((n, *self.latent_shape)).to(self.device)
        mode = self.decode(latent).argmax(dim=1).unsqueeze(1) / 255.0
        return mode
    
    def params(self, n_components, n_chunks=4):
        if n_components % n_chunks != 0:
            raise AttributeError("n_components should be divisible by n_chunks")
        
        chunk_size = n_components // n_chunks
        
        latents = Normal(0, 1).sample((n_components, *self.latent_shape)).to(self.device)
        params = torch.empty(n_components, 256, 28, 28).to(self.device)
        
        for i in range(n_chunks):
            r = range(i*chunk_size, (i+1)*chunk_size)
            params[r] = self.decode(latents[r])
        return { "probs": params, }
    
    def data_params(self, inputs):
        return { "probs": self.forward(inputs), }

        
        