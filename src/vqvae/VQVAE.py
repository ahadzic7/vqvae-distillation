import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tqdm import tqdm
from abc import abstractmethod
import torch.nn.functional as F
from src.vqvae.Embedding import VQEmbedding

class VQVAE(pl.LightningModule):
    def __init__(
            self, 
            encoder,
            decoder,
            input_shape,
            output_dim,
            latent_shape,
            codebook_size, 
            beta
        ):
        super(VQVAE, self).__init__()   
        
        def check_shape(expected, e, expected2, d, i_shape):
            with torch.no_grad():
                output = e(torch.zeros(128, *i_shape))
                output2 = d(output)
            shape = tuple(output.shape[1:])
            if shape != expected:
                raise Exception(...)
            shape = tuple(output2.shape[1:])
            if shape != expected2:
                raise Exception(...)
        
        ishape = (input_shape["channels"], input_shape["height"], input_shape["width"])
        lshape = (latent_shape["channels"], latent_shape["height"], latent_shape["width"])
        oshape = (output_dim, ishape[1], ishape[2])
        check_shape(lshape, encoder, oshape, decoder, ishape)
        
        self.encoder=encoder
        self.codebook=VQEmbedding(codebook_size, lshape[0])
        self.decoder=decoder
        self.beta=beta
        self.codebook_size=codebook_size
        self.latent_shape=lshape
        
        self.save_hyperparameters()

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)
    
    @torch.no_grad()
    def forward(self, x):
        z_e_x = self.encode(x)
        z_q_x = self.codebook(z_e_x)    
        z_w_x = self.codebook.embedding.weight[z_q_x].permute(0,3,1,2)
        return self.decode(z_w_x)
    
    def forward_loss(self, x):
        z_e_x = self.encode(x)
        z_q_x_st, z_q_x, distances = self.codebook.straight_through(z_e_x)    
        return self.decode(z_q_x_st), z_e_x, z_q_x, distances

    @abstractmethod
    def loss_recons(self, batch, out):
        """This method must be implemented by subclasses."""
        pass

    def loss_codebook(self, z_q_x, z_e_x):
        return F.mse_loss(z_q_x, z_e_x.detach())
    
    def loss_commitment(self, z_e_x, z_q_x):
        return F.mse_loss(z_e_x, z_q_x.detach())


    
    def loss(self, batch):
        """Compute all losses for VQ-VAE."""
        out, z_e_x, z_q_x, _ = self.forward_loss(batch)
        S = 1 # old loss function NLL
        # S = batch[0].numel() # NpD
        losses = {
            "rec": self.loss_recons(batch, out).mean() / S ,
            "cod": self.loss_codebook(z_q_x, z_e_x),
            "com": self.loss_commitment(z_e_x, z_q_x)
        }

        total_loss = losses["rec"] + losses["cod"] + self.beta * losses["com"]
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
        

    def codeword_utilization_frequency(self, loader):
        hist = torch.zeros(self.codebook_size).to(self.device)
        for x in tqdm(loader):
            z_e_x = self.encode(x.to(self.device))
            indices = self.codebook(z_e_x).squeeze(2).squeeze(1)
            hist += torch.bincount(indices, minlength=self.codebook_size)
        return hist

    def count_parameters(self, component_names=None):
        return sum(
            param.numel() for name, param in self.named_parameters()
            if component_names is None or any(name.startswith(cn) for cn in component_names)
        )
    

    def latent_cb_hists(self, datal):
        device = self.device
        H, W = self.latent_shape[1], self.latent_shape[2]
        hists = torch.zeros(H, W, self.codebook_size, dtype=torch.long, device=device)
        def proces(bx):
            nonlocal hists  # tell Python to use outer hists
            z_e_x = self.encode(bx.to(device))
            z_flat = self.codebook(z_e_x).view(-1, H*W).long()
            ones = torch.ones_like(z_flat, dtype=torch.long)
            hists_flat = hists.view(H*W, self.codebook_size)
            hists_flat.scatter_add_(1, z_flat.t(), ones.t())
            hists = hists_flat.view(H, W, self.codebook_size)

        bx = next(iter(datal))
        if isinstance(bx, torch.Tensor):
            for bx in tqdm(datal):
                proces(bx)
        else:
            for bx,_ in tqdm(datal):
                proces(bx)
        return hists
        

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