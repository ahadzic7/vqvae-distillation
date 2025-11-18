import torch
from torch.nn.functional import cross_entropy, log_softmax, softmax
from src.pcnn.PixelCNNBase import PixelCNNBase
from tqdm import tqdm
import torch.nn as nn  
from src.pcnn.masked_cnn.GatedMaskedConv2d import GatedMaskedConv2d
import sys

def batch_diagnostics(logits, latents):
    lm = logits.detach()
    print(f"logits min  {lm.min().item():.5f}")
    print(f"logits max  {lm.max().item():.5f}")
    print(f"logits mean {lm.mean().item():.5f}")
    print(f"logits std  {lm.std().item():.5f}")
    # check if any NaN or Inf
    print("has_nan:", torch.isnan(lm).any().item(), "has_inf:", torch.isinf(lm).any().item())
    # check target value range
    print("latents min/max:", latents.min().item(), latents.max().item())
    # compute sample cross-entropy if you like
    with torch.no_grad():
        preds = lm.argmax(dim=1)
        acc = (preds == latents).float().mean().item()
    print("batch top1 acc (random=~1/K):", acc)
    print()
    print()
    print()

class PixelCNN(PixelCNNBase):
    def __init__(self, vqvae, config):
        lat_dim = vqvae.latent_shape[0]
        embedding = nn.Embedding(vqvae.codebook_size, lat_dim)
        
        layers = nn.ModuleList()
        layers.append(GatedMaskedConv2d('A', lat_dim, 7, residual=False))
        for _ in range(config["n_layers"]-1):
            layers.append(GatedMaskedConv2d('B', lat_dim, 3, residual=True))

        output_conv = nn.Sequential(
            nn.Conv2d(lat_dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, vqvae.codebook_size, 1)
        )        
        super().__init__(vqvae, embedding, layers, output_conv)


    def forward(self, x):
        shp = x.size() + (-1,)
        x = self.embedding(x.view(-1)).view(shp)
        x = x.permute(0, 3, 1, 2)
        x_v, x_h = x, x
        for layer in self.layers:
            x_v, x_h = layer(x_v, x_h)
        return self.output_conv(x_h)
    
    def loss(self, batch_x):
        with torch.no_grad():
            z_e_x = self.vqvae.encode(batch_x).contiguous()
            latents = self.vqvae.codebook(z_e_x).detach()
        K = self.vqvae.codebook_size
        logits = self.forward(latents).permute(0, 2, 3, 1).contiguous().view(-1, K)
        latents = latents.view(-1)
        # batch_diagnostics(logits, latents)
        return cross_entropy(logits, latents)


    @torch.no_grad()
    def log_prob(self, z):
        logits = self.forward(z)
        log_probs = log_softmax(logits, dim=1)
        log_probs = log_probs.gather(dim=1, index=z.unsqueeze(1).long())
        return log_probs.sum(dim=[1, 2, 3])
    
    @torch.no_grad()
    def generate(self, n_samples, shape=(1, 1)):
        x = torch.zeros((n_samples, *shape), dtype=torch.int32, device=self.device)
        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self.forward(x)
                probs = softmax(logits[:, :, i, j], -1)
                x.data[:, i, j].copy_(probs.multinomial(1).squeeze().data)
        return x

    @torch.no_grad()
    def sample_latent(self, n_samples, batch_size=2**10):
        num_batches = -(-n_samples // batch_size)
        lat_shape = self.vqvae.latent_shape[1:]
        latents = torch.empty((n_samples, *lat_shape), dtype=torch.int32, device=self.device)

        for b in range(num_batches):
            start_id = b * batch_size
            end_id = min((b + 1) * batch_size, n_samples)
            gen = self.generate(n_samples=end_id-start_id, shape=lat_shape)
            latents[start_id:end_id] = gen
        return latents

    @torch.no_grad()
    def sample_unique_latent(self, n_samples, batch_size=2**10):
        unique_latents = []
        seen = set()
        progress_bar = tqdm(total=n_samples, desc="PixelCNN sampling")

        while len(unique_latents) < n_samples:
            old = len(unique_latents)
            for latent in self.sample_latent(n_samples=batch_size):
                latent_tuple = tuple(latent.view(-1).tolist())
                if latent_tuple in seen:
                    continue
                seen.add(latent_tuple)
                unique_latents.append(latent)
                if len(unique_latents) == n_samples:
                    break
            progress_bar.update(len(unique_latents) - old)
        unique_latents = torch.stack(unique_latents, dim=0)
        return unique_latents

    @torch.no_grad()
    def _params(self, n_components, unique=True, batch_size=2**20):
        _, H, W = self.vqvae.latent_shape
        max_comp = self.vqvae.codebook_size ** (H * W)
        MAX_FLOAT = sys.float_info.max

        # if not (n_components <= MAX_FLOAT / 20 and max_comp <= MAX_FLOAT / 3):
        #     raise Exception(f"Number of components not possible to compute!")
        
        weight = None
        if 20 * n_components >= 3 * max_comp:
            latent = torch.cat([z for z in tqdm(self.cartesian_product_batch(batch_size))], dim=0)
            if n_components >= max_comp:
                print("Number of components exceeds latent space; reducing to max ", max_comp)
            elif n_components >= (max_comp / 20) * 3:
                print(f"{n_components} >= {3/20*100}% of latent space -> selecting top {n_components} by p(z)")
                weight_all = torch.cat([self.log_prob(z) for z in latent.split(batch_size, dim=0)])
                weight, indices = torch.topk(weight_all, n_components)
                latent = latent[indices]
        else:
            print("Unique sampling" if unique else "Non-unique sampling")
            sampling = self.sample_unique_latent if unique else self.sample_latent
            latent = sampling(n_components)
            print("Number of unique 3D tensors:", latent.unique(dim=0).size(0))

        return latent, weight

    @torch.no_grad()
    def params(self, n_components, unique=True, batch_size=2**20):
        latent, _ = self._params(n_components, unique, batch_size)
        return self.vqvae.params(latent)

    @torch.no_grad()
    def params_w(self, n_components, unique=True, batch_size=2**20):
        latent, weight = self._params(n_components, unique, batch_size)
        if weight is None:
            weight = torch.cat([self.log_prob(z) for z in latent.split(batch_size, dim=0)])
        return self.vqvae.params(latent), weight

    @torch.no_grad()
    def sample(self, n_samples, unique=True):
        latent = self.sample_latent(n_samples, unique=unique)
        return self.vqvae.sample(latent)

    def training_step(self, batch, batch_id):
        loss = self.loss(batch)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_id):
        loss = self.loss(batch)
        self.log('valid_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss