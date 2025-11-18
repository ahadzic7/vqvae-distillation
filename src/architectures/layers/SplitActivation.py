import torch
import torch.nn as nn
import torch.nn.functional as F

class SplitActivation(nn.Module):
    def __init__(self, fc1, min_max_clamp):
        super(SplitActivation, self).__init__()
        self.fc1 = fc1
        self.min=min_max_clamp[0]
        self.max=min_max_clamp[1]

    def forward(self, input): 
        mu, log_var = input.chunk(2, dim=1)
        dev = log_var.device
        clv = torch.clamp(log_var, min=self.min.to(dev), max=self.max.to(dev))
        self.max = self.max.to(dev)
        self.min = self.min.to(dev)

        return torch.cat((self.fc1(mu), clv), dim=1)
    
    @classmethod
    def setup(cls, min_max_std, mu_act, output_dim):
        if min_max_std is None and output_dim % 2 != 0:
            raise AttributeError(f"If learn std is {min_max_std}, num of channels must be 1!")

        min_std, max_std = min_max_std
        min_log_var = 2 * torch.log(torch.tensor(min_std))
        max_log_var = 2 * torch.log(torch.tensor(max_std))
        mu_activations = {
            "tanh": F.tanh,
            "sigmoid": F.sigmoid,
        }

        return cls(
            fc1=mu_activations[mu_act],
            min_max_clamp=(min_log_var, max_log_var),
        )