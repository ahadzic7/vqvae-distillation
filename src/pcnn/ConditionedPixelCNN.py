import torch
from torch.nn.functional import cross_entropy, log_softmax, softmax
from src.pcnn.PixelCNNBase import PixelCNNBase
from tqdm import tqdm
import torch.nn as nn  
from src.pcnn.masked_cnn.ConditionedGatedMaskedConv2d import ConditionedGatedMaskedConv2d

class ConditionedPixelCNN(PixelCNNBase):
    def __init__(self, vqvae, classes, config):
        lat_dim = vqvae.latent_shape[0]
        embedding = nn.Embedding(vqvae.codebook_size, lat_dim)
        
        layers = nn.ModuleList()
        layers.append(ConditionedGatedMaskedConv2d('A', lat_dim, 7, classes.size(0), residual=False))
        for _ in range(config["n_layers"]-1):
            layers.append(ConditionedGatedMaskedConv2d('B', lat_dim, 3, classes.size(0), residual=True))

        output_conv = nn.Sequential(
            nn.Conv2d(lat_dim, 512, 1),
            nn.ReLU(True),
            nn.Conv2d(512, vqvae.codebook_size, 1)
        )
        
        super().__init__(vqvae, embedding, layers, output_conv)
        self.classes = classes

    def forward(self, x, label):
        shp = x.size() + (-1,)
        x = self.embedding(x.view(-1)).view(shp)
        x = x.permute(0, 3, 1, 2)
        x_v, x_h = x, x
        for layer in self.layers:
            x_v, x_h = layer(x_v, x_h, label)  # uses label
        return self.output_conv(x_h)
    
    def loss(self, batch_x, batch_y):
        with torch.no_grad():
            z_e_x = self.vqvae.encode(batch_x).contiguous()
            latents = self.vqvae.codebook(z_e_x).detach()
        K = self.vqvae.codebook_size
        logits = self.forward(latents, batch_y).permute(0, 2, 3, 1).contiguous()
        return cross_entropy(logits.view(-1, K), latents.view(-1))

    @torch.no_grad()
    def log_prob(self, z, label):
        logits = self.forward(z, label)
        log_probs = log_softmax(logits, dim=1)
        log_probs = log_probs.gather(dim=1, index=z.unsqueeze(1).long())
        return log_probs.sum(dim=[1, 2, 3])

    @torch.no_grad()
    def mar_log_prob(self, z, batch_size=2**14):
        ll_pc = torch.log(torch.tensor(self.classes.size(0)))
        lls_list = []
        for z_batch in torch.split(z, batch_size, dim=0):
            lls_batch = [self.log_prob(z_batch, c) - ll_pc for c in self.classes.view(-1, 1).to(self.device)]
            lls_list.append(torch.stack(lls_batch, dim=0).logsumexp(dim=0))
        return torch.cat(lls_list, dim=0)

    
    @torch.no_grad()
    def generate(self, label=None, shape=(1, 1)):
        x = torch.zeros(
            (label.shape[0] if label is not None else 1, *shape),
            dtype=torch.int64,
            device=self.device
        )
        for i in range(shape[0]):
            for j in range(shape[1]):
                logits = self.forward(x, label)
                probs = softmax(logits[:, :, i, j], -1)
                x.data[:, i, j].copy_(probs.multinomial(1).squeeze().data)
        return x

    @torch.no_grad()
    def sample_latent(self, n_samples=None, labels=None, batch_size=2**10):
        if labels is None:
            random_indices = torch.randint(0, len(self.classes), (n_samples,))
            labels = self.classes[random_indices].to(self.device)

        num_batches = -(-labels.shape[0] // batch_size)
        lshape = self.vqvae.latent_shape[1:]
        latents = torch.empty((labels.shape[0], *lshape), device=self.device, dtype=torch.int)

        for b in range(num_batches):
            start_id = b * batch_size
            end_id = min((b + 1) * batch_size, labels.shape[0])
            batch_labels = labels[start_id:end_id]
            latents[start_id:end_id] = self.generate(batch_labels, lshape)
        return latents

    @torch.no_grad()
    def sample_unique_latent(self, n_samples=None, labels=None, batch_size=2**10):
        if n_samples is None:
            n_samples = labels.shape[0]
        unique_latents = []
        seen = set()
        progress_bar = tqdm(total=n_samples, desc="PixelCNN sampling")

        while len(unique_latents) < n_samples:
            old = len(unique_latents)
            for latent in self.sample_latent(n_samples=batch_size, labels=labels, batch_size=batch_size):
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
    def _params(self, n_components=None, labels=None, unique=True, batch_size=2**20):
        assert n_components is not None or labels is not None, \
            "Number of components or labels should be defined"

        _, H, W = self.vqvae.latent_shape
        max_comp = self.vqvae.codebook_size ** (H * W)
        sampling = self.sample_unique_latent if unique else self.sample_latent
        weight = None

        if n_components is None:
            print("Sampling by labels")
            latent = sampling(labels=labels)
        elif n_components >= max_comp:
            print("Number of components exceeds latent space; reducing to max ", max_comp)
            latent = torch.cat([z for z in tqdm(self.cartesian_product_batch(batch_size))], dim=0)
            if n_components >= (max_comp / 20) * 3:
                print(f"{n_components} >= {3/20*100}% of latent space -> selecting top {n_components}")
                latent_all = latent
                weight_all = torch.cat([self.mar_log_prob(z) for z in latent_all.split(batch_size, dim=0)])
                weight, indices = torch.topk(weight_all, max_comp)
                latent = latent_all[indices]
        else:
            print("Unique sampling" if unique else "Non-unique sampling")
            latent = sampling(n_samples=n_components)
            print("Number of unique 3D tensors:", latent.unique(dim=0).size(0))

        return latent, weight

    @torch.no_grad()
    def params(self, n_components=None, labels=None, unique=True, batch_size=2**20):
        latent, _ = self._params(n_components, labels, unique, batch_size)
        return self.vqvae.params(latent)

    @torch.no_grad()
    def params_w(self, n_components=None, labels=None, unique=True, batch_size=2**20):
        latent, weight = self._params(n_components, labels, unique, batch_size)
        if weight is None:
            weight = torch.cat([self.mar_log_prob(z) for z in latent.split(batch_size, dim=0)])
        return self.vqvae.params(latent), weight

    @torch.no_grad()
    def sample(self, label_range, n_samples=None, labels=None, unique=True):
        latent = self.sample_latent(label_range, n_samples=n_samples, labels=labels, unique=unique)
        return self.vqvae.sample(latent)

    def training_step(self, batch, batch_id):
        loss = self.loss(*batch)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_id):
        loss = self.loss(*batch)
        self.log('valid_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss