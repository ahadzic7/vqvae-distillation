import torch
from src.mm.GaussianMixture import GaussianMixture
from src.cm.ContinuousMixture import ContinuousMixture
from src.EinsumNetwork.EinsumNetwork import EinsumNetwork

class SamplerDataset(torch.utils.data.Dataset):
    """Dataset that generates samples on-demand from a sampler."""
    
    def __init__(self, sampler, total_samples, dims=None):
        self.sampler = sampler
        self.total_samples = total_samples

    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, _):
        return self.sampler.sample(1).squeeze(0)