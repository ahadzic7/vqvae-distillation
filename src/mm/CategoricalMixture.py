import torch
from torch.distributions import Categorical, OneHotCategorical
from src.mm.MixtureModel import MixtureModel
from tqdm import tqdm

class CategoricalMixture(MixtureModel):
    def __init__(self, theta, mixing_weights=None, ohe=False, batching=False):
        if ohe:
            probs = theta["probs"].permute(0,2,3,1).unsqueeze(0)
        else:
            probs = theta["probs"].permute(0,2,3,1).unsqueeze(0)
        super().__init__(
            mixing_weights=mixing_weights,
            n_components = probs.shape[1],
            batching=batching
        )
        self.ohe = ohe
        self.dt = OneHotCategorical if self.ohe else Categorical
        self.dims = [2, 3] if self.ohe else [1, 3, 4]
        
        self.n_components = probs.shape[1]
        self.shape = tuple(probs.shape[2:])
        self.batch_size=2**13

        self.batches = torch.split(probs, self.batch_size, dim=1)

    def _lls(self, x):
        x = x.unsqueeze(1)
        if self.ohe:
            x = x.permute(0, 1, 3, 4, 2)
        return torch.cat([self.dt(probs=p).log_prob(x).sum(dim=self.dims) for p in self.batches], dim=1)

    @torch.no_grad()
    def log_prob(self, x):
        return super(CategoricalMixture, self).logsumexp(self._lls(x))

    @torch.no_grad()
    def max_ll_component(self, x):
        return torch.argmax(self._lls(x), dim=1)

    @torch.no_grad()
    def _sampling(self, components):
        samples = []
        for c in components:
            batch_idx, component_idx = divmod(c.item(), self.batch_size)
            d = self.dt(probs=self.batches[batch_idx][:, component_idx])
            samples.append(d.sample((1,)))
        samples = torch.cat(samples, dim=0)
        if self.ohe:
            samples = samples.squeeze(1).permute(0, 3, 1, 2)#.argmax(dim=1, keepdim=True)
        return samples

    @torch.no_grad()
    def bpd(self, x, f=lambda x: x):
        mm = super(CategoricalMixture, self)
        return mm.bpd(x, f)

    @torch.no_grad()
    def sample(self, n_samples=0, components=None):
        if components is None:
            components = self.sample_component(n_samples)
        return self._sampling(components)

    @torch.no_grad()
    def inpaint(self, x, mask):
        mask = mask.bool()
        if not (mask[:, 0] == 0).any():
            return x

        x_inpaint = x.clone()
        x = x.unsqueeze(1)
        if self.ohe:
            x = x.permute(0, 1, 3, 4, 2)
        processed_mask = mask if self.ohe else mask.unsqueeze(1)

        lls = [self.dt(probs=p).log_prob(x) *  processed_mask for p in self.batches]
        lls = torch.cat([ll.sum(dim=self.dims) for ll in lls], dim=1)
        posterior = torch.softmax(lls + torch.log(self.mixing_weights).unsqueeze(0), dim=1) 
        row_sample = lambda p: torch.multinomial(p, 1, replacement=True)
        comp_indices = torch.vmap(row_sample, randomness='different')(posterior).reshape(-1)
        
        samples = self._sampling(comp_indices)
        x_inpaint = x_inpaint * mask + samples * ~mask
        return x_inpaint
    

    @torch.no_grad()
    def total_parameters(self):
        H, W, C = self.shape
        return ((C-1) * H * W + 1) * self.n_components - 1
    
    def save(self, filename="categorical_mix_model.mm"):
        torch.save(self.state_dict(), filename)
