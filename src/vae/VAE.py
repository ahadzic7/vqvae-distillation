import torch
import pytorch_lightning as pl
from torch.distributions import Normal, kl_divergence
from torch.utils.data import DataLoader
from tqdm import tqdm
from abc import abstractmethod
import math


import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm
from abc import abstractmethod

class VAE(pl.LightningModule):
    def __init__(
            self, 
            encoder,
            decoder,
            input_shape,
            output_dim,
            latent_shape,
            beta
        ):
        super(VAE, self).__init__()   
        def check_shape(expected, e, expected2, d, i_shape):
            e = e.eval()
            d = d.eval()
            with torch.no_grad():
                output = e(torch.zeros(1, *i_shape)).chunk(2, dim=1)[0]
                output2 = d(output)
            shape = tuple(output.shape[1:])
            
            if shape != expected:
                raise Exception(f"Shapes missmatch on encoder -> expected {expected}, but got {shape}")
            shape = tuple(output2.shape[1:])
            if shape != expected2:
                raise Exception(f"Shapes missmatch on decoder -> expected {expected2}, but got {shape}")
            e = e.train()
            d = d.train()
        
        ishape = (input_shape["channels"], input_shape["height"], input_shape["width"])
        lshape = (latent_shape["channels"]//2, latent_shape["height"], latent_shape["width"])
        oshape = (output_dim, ishape[1], ishape[2])

        check_shape(lshape, encoder, oshape, decoder, ishape)
        
        self.encoder=encoder
        self.decoder=decoder
        self.beta=beta
        self.latent_shape=lshape
        
        self.L2_PI = torch.tensor(math.log(2*math.pi)).to("cuda")

        self.save_hyperparameters()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)


    @torch.no_grad()
    def forward(self, x:torch.Tensor):
        mu, log_var = self.encode(x).chunk(2, dim=1)
        z = mu + log_var.mul(0.5).exp() * torch.randn_like(mu)
        return self.decode(z)
    
    def forward_loss(self, x:torch.Tensor):
        mu, log_var = self.encode(x).chunk(2, dim=1)
        # KL divergence between q(z|x) ~ N(mu, std^2) and p(z) ~ N(0, 1)
        kl_div = 0.5 * (log_var.exp() + mu.pow(2) - 1 - log_var).sum(1).mean()
        z = mu + log_var.mul(0.5).exp() * torch.randn_like(mu)
        return self.decode(z), kl_div


    @abstractmethod
    def loss_recons(self, batch, out):
        """This method must be implemented by subclasses."""
        pass

    
    def loss(self, batch):
        """Compute all losses for VQ-VAE."""
        out, kl_div = self.forward_loss(batch)
        S = 1 # old loss function NLL
        # S = batch[0].numel() # NpD
        losses = {
            "rec": self.loss_recons(batch, out).mean() / S ,
            "kl_div": kl_div
        }
        total_loss = losses["rec"] + self.beta * losses["kl_div"]
        return total_loss, losses

    def training_step(self, batch: torch.Tensor, batch_idx: int):
        """Training step with consistent loss logging."""
        self.train()
        total_loss, component_losses = self.loss(batch)

        # Log total loss
        self.log('train_loss', total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        # Log component losses with consistent naming
        for component_name, loss_value in component_losses.items():
            self.log(f"train_{component_name}", loss_value, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return total_loss
    
    @torch.no_grad()
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        """Validation step with consistent loss logging."""
        self.eval()
        total_loss, component_losses = self.loss(batch)

        # Use 'val_' prefix for consistency with the logger
        self.log('valid_loss', total_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        # Log component losses with consistent naming
        for component_name, loss_value in component_losses.items():
            self.log(f"valid_{component_name}", loss_value, 
                    prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return total_loss



    ###
    # sampling the prior
    @torch.no_grad()
    def _params(self, n_components, unique=True, batch_size=2**10):
        print("Sampling")
        mu = torch.zeros((n_components, *self.latent_shape)).to(self.device)
        sd = torch.ones((n_components, *self.latent_shape)).to(self.device)
        latent = mu + sd * torch.randn_like(mu)
        weight = -0.5 * (latent.pow(2) + self.L2_PI).sum(dim=[1,2,3])
        return latent, weight




    @torch.no_grad()
    def eval_loader(
        self, 
        loader:DataLoader, 
        progress_bar:bool = False, 
        device:str = 'cpu'
    ):
        self.eval()
        loader = tqdm(loader) if progress_bar else loader
        return torch.cat([self.loss(x.to(device), *self.forward_loss(x.to(device))) for x in loader], dim=0)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)

    def count_parameters(self, component_names=None):
        return sum(
            param.numel() for name, param in self.named_parameters()
            if component_names is None or any(name.startswith(cn) for cn in component_names)
        )
        

    @abstractmethod
    def reconstruction(self, x:torch.Tensor):
        """This method must be implemented by subclasses."""
        pass

    @abstractmethod
    def sample(self, n_samples):
        """This method must be implemented by subclasses."""
        pass

    @abstractmethod
    def params(self, latent):
        """This method must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def data_params(self, inputs):
        """This method must be implemented by subclasses."""
        pass

    @abstractmethod
    def uniform_params(self, n_components):
        """This method must be implemented by subclasses."""
        pass
