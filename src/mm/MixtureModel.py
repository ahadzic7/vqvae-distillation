import torch
from torch.distributions import Categorical
import torch.nn.functional as F
from abc import abstractmethod
from tqdm import tqdm

class MixtureModel(torch.nn.Module):
    def __init__(self, mixing_weights, batching, n_components=0):
        super(MixtureModel, self).__init__()

        if mixing_weights is None and n_components > 0:
            mw = torch.ones(n_components) / float(n_components)
        elif mixing_weights is not None and batching:
            mw = mixing_weights
        elif mixing_weights is not None:
            s = mixing_weights.sum()
            mw = mixing_weights / s
        # print(mw)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'                
        self.n_components = mw.size(0)
        self.mixing_weights = mw.to(self.device)
        self.mixing_weight_dist = Categorical(probs=self.mixing_weights)

    def sample_component(self, n_samples):
        return self.mixing_weight_dist.sample((n_samples,))

    @torch.no_grad()
    def logsumexp(self, lls):
        log_w = torch.log(self.mixing_weights).unsqueeze(0)
        return torch.logsumexp(lls + log_w, dim=1)
        
    
    @torch.no_grad()
    def bpd(self, x, f):
        D = torch.prod(torch.tensor(x.shape[1:]))
        bpd = -self.log_prob(x).div(D * torch.log(torch.tensor(2)))
        return f(bpd) 
    
    def max_ll_component_frequency(self, loader, progress=False): 
        hist = torch.zeros(self.mixing_weights.shape[0]).to(self.device)
        loader = tqdm(loader) if progress else loader
        for bx in loader: 
            indices = self.max_ll_component(bx.to(self.device))
            hist += torch.bincount(indices, minlength=self.n_components)
        return hist

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        self.load_state_dict(torch.load(filename))

    @abstractmethod
    def sample(self, n_samples):
        pass

    @abstractmethod
    def max_ll_component(self, x, dim):
        pass

    

    