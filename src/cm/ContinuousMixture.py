import torch
from torch.utils.data import DataLoader
from typing import Callable, Optional
import pytorch_lightning as pl
from tqdm import tqdm
import torch.nn as nn
from torch.distributions import OneHotCategorical, Categorical
import torch.nn.functional as F
from src.cm.decoders import GaussianDecoder, CategoricalDecoder

class ContinuousMixture(pl.LightningModule):
    def __init__(
            self, 
            decoder: nn.Module, 
            sampler: Callable, 
            k: Optional[int] = None, 
            n_samples=2**14,
            lat_dim=16
        ):
        super(ContinuousMixture, self).__init__()
        self.decoder = decoder
        self.sampler = sampler
        self.k = k
        self.n_chunks = None
        self.missing = None
        self.n_samples = n_samples
        self.lat_dim = lat_dim
        self.save_hyperparameters(ignore=['n_chunks', 'missing'])
    
    def _forward(self, x, z, log_w, k, seed):
        assert (z is None and log_w is None) or (z is not None and log_w is not None)
        
        if z is None:
            z, log_w = self.sampler(seed=seed)
            # use when decoder is from CM
            # z = torch.randn(self.n_samples, self.lat_dim, device=self.device)

        z = z.to(x.device)
        log_w = log_w.to(x.device)
        log_prob_bins = self.decoder(x, log_w, z, k, self.missing, self.n_chunks)
        assert log_prob_bins.size() == (x.size(0), z.size(0)) or log_prob_bins.size() == (x.size(0), k)
        return log_prob_bins

    def forward(
        self,
        x,
        z = None,
        log_w= None,
        k = None,
        seed= None
    ):
        log_prob_bins = self._forward(x, z, log_w, k, seed)
        return torch.logsumexp(log_prob_bins, dim=1, keepdim=False)

    @torch.no_grad()
    def log_prob(self, x):
        return self.forward(x)

    @torch.no_grad()
    def logits(self, x, z, log_w, k, seed):
        return self._forward(x, z, log_w, k, seed)

    @torch.no_grad()
    def max_log_prob_component(self, x, z, log_w, k, seed):
        return self._forward(x, z, log_w, k, seed).argmax(dim=1, keepdim=False)
    
    @torch.no_grad()
    def sample(self, n_samples, channels=256):
        z, log_w = self.sampler(seed=None)
        components = Categorical(logits=log_w).sample((n_samples,))

        decoded = []
        if isinstance(self.decoder, CategoricalDecoder):
            z_sel = z[components].to(self.device)  # batch index selection
            logits = self.decoder.decode_latent(z_sel, self.n_chunks)  # shape: (n_samples, ...)
            decoded = F.softmax(logits, dim=1).argmax(dim=1, keepdim=True)  # (n_samples, 1, H, W)

        elif isinstance(self.decoder, GaussianDecoder):
            z = z.to(self.device)
            decoded = self.decoder.sample_continuous(z, components)
            decoded = decoded.float().to(self.device)

        return decoded


    @torch.no_grad()
    def bpd(self, x, f=None):
        if f is None:
            f = lambda b: b
        d = x.shape[2] * x.shape[3]
        bpd = -self.log_prob(x).div(d).div(torch.log(torch.tensor(2)))
        return f(bpd)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
    
    def total_parameters(self, feature_dim):
        if type(self.decoder) is CategoricalDecoder:
            return self.sampler.n_bins - 1 + 255 * (self.sampler.n_bins * feature_dim)
        elif type(self.decoder) is GaussianDecoder:
            return (self.sampler.n_bins - 1) + 2 * (self.sampler.n_bins * feature_dim)
    
    @torch.no_grad()
    def inpaint(self, x, mask):
        missing_idx = (mask[:, 0] == 0).nonzero(as_tuple=False)
        if missing_idx.shape[0] == 0:
            return x
        
        x = x.float().masked_fill_(~mask.bool(), float('nan'))

        z, log_w = self.sampler(seed=None)
        z = z.to(self.device)
        self.missing = True
        logits = self.logits(x, z=z, log_w=log_w)
        posterior = torch.softmax(logits, dim=1)
        self.missing = False

        components = torch.multinomial(posterior, num_samples=1, replacement=True).to(self.device)        
        sampled_z = z[components.squeeze(1)]
        
        if type(self.decoder) is CategoricalDecoder:
            logits = self.decoder.decode_latent(sampled_z)[0].log_softmax(dim=1).permute(0, 2, 3, 1)
            samples = Categorical(logits=logits).sample().unsqueeze(1)
        else:
            samples = self.decoder.sample_continuous(z=sampled_z).float()

        x_inpaint = torch.where(torch.isnan(x), samples, x)
        if type(self.decoder) is CategoricalDecoder:
            x_inpaint /= 255.
        
        return x_inpaint.clamp(min=0, max=1)
    
    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


    def training_step(self, batch, batch_idx):
        # Compute log-likelihood of the batch, Loss is negative log likelihood
        loss = -self.forward(batch, k=self.k).mean()
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss
    
    @torch.no_grad()
    def validation_step(self, batch: torch.Tensor, batch_idx: int):
        loss = -self.forward(batch, k=None, seed=42).mean()
        self.log('valid_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    @torch.no_grad()
    def eval_loader(
        self, 
        loader: DataLoader, 
        z: Optional[torch.Tensor] = None,  
        log_w: Optional[torch.Tensor] = None,      
        seed: Optional[int] = None, 
        progress_bar: Optional[bool] = False, 
        device: str = 'cuda'
    ):
        self.eval()
        loader = tqdm(loader) if progress_bar else loader
        lls = torch.cat([self.forward(x.to(device), z, log_w, k=None, seed=seed) for x in loader], dim=0)
        # assert len(lls) == len(loader.dataset)
        return lls
