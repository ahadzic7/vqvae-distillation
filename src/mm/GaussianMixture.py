import torch
from src.mm.MixtureModel import MixtureModel

class GaussianMixture(MixtureModel):
    def __init__(self, theta, mixing_weights=None, batching=False):
        mean, sd = theta["mean"], theta["sd"]
        if mean.shape != sd.shape:
            raise ValueError("Mean and SD tensors must have the same shape.")
        super().__init__(
            mixing_weights=mixing_weights,
            n_components = mean.shape[0],
            batching=batching
        )
        self.n_components = mean.shape[0]
        self.shape = tuple(mean.shape[1:])
        self.batch_size=2**3
        self.L_2_PI=torch.log(torch.tensor(2 * torch.pi))
        
        mean_batches = torch.split(mean.unsqueeze(0), self.batch_size, dim=1)
        sd_batches = torch.split(sd.unsqueeze(0), self.batch_size, dim=1)
        self.batches = list(zip(mean_batches, sd_batches))
    
    @torch.no_grad()
    def normal_log_prob(self, x, mu, sd):
        """Compute log probability of x under Normal(mu, sd)."""
        return -0.5 * (((x - mu) ** 2) / (sd ** 2) + 2 * torch.log(sd) + self.L_2_PI)
    
    @torch.no_grad()
    def normal_sample(self, mu, sd, component_idx=None, clamp=True):
        """
            Sample from a Normal(mu, sd) distribution with optional component indexing and clamping.

            Args:
                mu (torch.Tensor): Mean tensor of shape (batch, num_components) or (batch,).
                sd (torch.Tensor): Std tensor of same shape as mu.
                component_idx (int or Tensor, optional): Component index or indices to select.
                clamp (bool): Whether to clamp samples to [0, 1].

            Returns:
                torch.Tensor: Sampled tensor of shape (batch,).
        """
        if component_idx is not None:
            mu = mu[:, component_idx]
            sd = sd[:, component_idx]
        smp = mu + sd * torch.randn_like(mu)
        return smp if not clamp else smp.clamp(0.0, 1.0)

    @torch.no_grad()
    def log_prob(self, x):
        x = x.unsqueeze(1)
        lls = torch.cat([self.normal_log_prob(x, mu, sd).sum(dim=[2,3,4]) for mu, sd in self.batches], dim=1)
        return super(GaussianMixture, self).logsumexp(lls)
         
    @torch.no_grad()
    def max_ll_component(self, x):
        x = x.unsqueeze(1)
        lls = torch.cat([self.normal_log_prob(x, mu, sd).sum(dim=[2,3,4]) for mu, sd in self.batches], dim=1)
        return torch.argmax(self.mixing_weights.log() + lls, dim=1)
    
    @torch.no_grad()
    def _sampling(self, components):
        samples = []
        for c in components:
            batch_idx, component_idx = divmod(c.item(), self.batch_size)
            mu, sd = self.batches[batch_idx]
            smp = self.normal_sample(mu, sd, component_idx, clamp=True)
            #d = Normal(mu[:, component_idx], sd[:, component_idx])
            #smp = d.sample((1,))[:, 0] # I think this should be here .clamp(0.0, 1.0)
            samples.append(smp)
        return torch.cat(samples, dim=0)
            
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

        lls = [self.normal_log_prob(x.unsqueeze(1), mu, sd) for mu, sd in self.batches]
        lls = torch.cat([(ll *  mask.unsqueeze(1)).sum(dim=[2, 3, 4]) for ll in lls], dim=1)
        posterior = torch.softmax(lls + torch.log(self.mixing_weights).unsqueeze(0), dim=1) 
        row_sample = lambda p: torch.multinomial(p, 1, replacement=True)
        comp_indices = torch.vmap(row_sample, randomness='different')(posterior).reshape(-1)
        
        samples = self._sampling(comp_indices)
        x_inpaint = x_inpaint * mask + samples * ~mask
        return torch.clamp(x_inpaint, min=0., max=1.)
    
    @torch.no_grad()
    def total_parameters(self):        
        C, H, W = self.shape
        return (2 * C * H * W + 1) * self.n_components - 1