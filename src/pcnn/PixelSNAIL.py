import torch
from torch.nn.functional import cross_entropy, log_softmax, softmax
from src.pcnn.PixelCNNBase import PixelCNNBase
from tqdm import tqdm
import torch.nn as nn  
from src.pcnn.masked_cnn.GatedMaskedConv2d import GatedMaskedConv2d
import sys
from src.architectures.layers.CausalAttention import CausalAttention
from src.architectures.layers.CausalConv2d import CausalConv2d
from src.architectures.layers.CausalResBlock import CausalResBlock
from src.architectures.layers.AttentionBlock import AttentionBlock

def batch_diagnostics(logits, latents, model=None):
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

class PixelSNAIL(PixelCNNBase):
    def __init__(self, vqvae, config):
        """
        config should contain:
        - n_layers: number of residual blocks
        - n_heads: number of attention heads
        - hidden_dim: hidden dimension (default: 128)
        - attn_every: insert attention every N layers (default: 4)
        - dropout: dropout rate (default: 0.1)
        """
        lat_dim = config.get("hidden_dim", 128)
        n_layers = config["n_layers"]
        n_heads = config.get("n_heads", 4)
        attn_every = config.get("attn_every", 4)
        dropout = config.get("dropout", 0.1)
        
        embedding = nn.Embedding(vqvae.codebook_size, lat_dim)
        
        layers = nn.ModuleList()
        # Initial causal convolution
        layers.append(CausalConv2d(lat_dim, lat_dim, kernel_size=7, mask_type='A'))
        
        # Interleave residual blocks and attention blocks
        for i in range(n_layers):
            layers.append(CausalResBlock(lat_dim, dropout=dropout))
            if (i + 1) % attn_every == 0:
                layers.append(AttentionBlock(lat_dim, n_heads, dropout))
        
        output_conv = nn.Sequential(
            nn.LayerNorm([lat_dim]),
            nn.Conv2d(lat_dim, 512, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(512, vqvae.codebook_size, 1)
        )
        
        super().__init__(vqvae, embedding, layers, output_conv)
        
    def forward(self, x):
        shp = x.size() + (-1,)
        x = self.embedding(x.view(-1)).view(shp)
        x = x.permute(0, 3, 1, 2)
        
        for layer in self.layers:
            x = layer(x)

        # Handle LayerNorm in output_conv
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = self.output_conv[0](x)  # LayerNorm
        x = x.permute(0, 3, 1, 2)  # Back to (B, C, H, W)
        
        for layer in self.output_conv[1:]:
            x = layer(x)
        
        return x
    
    def loss(self, batch_x):
        with torch.no_grad():
            z_e_x = self.vqvae.encode(batch_x).contiguous()
            latents = self.vqvae.codebook(z_e_x).detach()
        K = self.vqvae.codebook_size
        logits = self.forward(latents).permute(0, 2, 3, 1).contiguous().view(-1, K)
        logits = torch.randn_like(logits, requires_grad=True)
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
                probs = softmax(self.forward(x)[:, :, i, j], -1)
                samples = probs.multinomial(1).squeeze().data
                x.data[:, i, j].copy_(samples)
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
        return latents.unique(dim=0)

    @torch.no_grad()
    def sample_unique_latent(self, n_samples, batch_size=2**10):
        seen = set()
        progress_bar = tqdm(total=n_samples, desc="PixelSNAIL sampling")

        while len(seen) < n_samples:
            old_len = len(seen)
            for latent in self.sample_latent(n_samples=batch_size):
                latent_tuple = tuple(latent.view(-1).tolist())
                seen.add(latent_tuple)
                if len(seen) >= n_samples:
                    break
            progress_bar.update(len(seen) - old_len)

        # Convert the set of tuples back to a tensor
        ul = [torch.tensor(t).view(latent.shape) for t in list(seen)[:n_samples]]
        return torch.stack(ul, dim=0)
    
    @torch.no_grad()
    def _params(self, n_components, unique=True, batch_size=2**20):
        _, H, W = self.vqvae.latent_shape
        max_comp = self.vqvae.codebook_size ** (H * W)
        MAX_FLOAT = sys.float_info.max
        
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
    
    # def on_after_backward(self):
        # Called after loss.backward() but before optimizer.step()
        # self.analyze_gradients()

    def analyze_gradients(self, tiny_threshold=1e-6, large_threshold=10.0):
        """
        Analyze model gradients and print warnings if something is wrong.

        Args:
            self: nn.Module
            tiny_threshold: below this mean abs grad is considered too small
            large_threshold: above this mean abs grad is considered too large
        """
        any_grad = False
        issues = []

        for name, param in self.named_parameters():
            if param.grad is None:
                continue

            any_grad = True
            grad_mean = param.grad.abs().mean().item()

            if grad_mean == 0.0:
                issues.append(f"{name}: GRAD IS EXACTLY ZERO")
            elif grad_mean < tiny_threshold:
                issues.append(f"{name}: GRAD VERY SMALL ({grad_mean:.2e})")
            elif grad_mean > large_threshold:
                issues.append(f"{name}: GRAD VERY LARGE ({grad_mean:.2e})")

        if not any_grad:
            print("⚠️ No gradients found in the model! Check backward pass.")
        elif issues:
            print("⚠️ Gradient warnings:")
            for i in issues:
                print("   ", i)
        else:
            print("✅ Gradients look healthy.")