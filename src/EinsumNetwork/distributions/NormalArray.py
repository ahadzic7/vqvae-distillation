import torch 
from src.EinsumNetwork.FactorizedLeafLayer import ExponentialFamilyArray, shift_last_axis_to

class NormalArray(ExponentialFamilyArray):
    """Implementation of Normal distribution."""

    # def __init__(self, num_var, num_dims, array_shape, min_var=0.0001, max_var=10., use_em=True):
    def __init__(self, num_var, num_dims, array_shape, min_std, max_std, use_em=True):
        super(NormalArray, self).__init__(num_var, num_dims, array_shape, 2 * num_dims, use_em=use_em)
        self.log_2pi = torch.tensor(1.8378770664093453)
        min_var = min_std**2
        max_var = max_std**2
        print(f"std in {min_std, max_std}")
        print(f"var in {min_var, max_var}")
        self.min_var = min_var
        self.max_var = max_var

    def default_initializer(self):
        phi = torch.empty(self.num_var, *self.array_shape, 2*self.num_dims)
        
        with torch.no_grad():
            phi[..., 0:self.num_dims] = torch.randn(self.num_var, *self.array_shape, self.num_dims)
            phi[..., self.num_dims:] = 1. + phi[..., 0:self.num_dims]**2
        return phi

    def project_params(self, phi):
        phi_project = phi.clone()
        mu2 = phi_project[..., 0:self.num_dims] ** 2
        phi_project[..., self.num_dims:] -= mu2
        
        var = phi_project[..., self.num_dims:]
        phi_project[..., self.num_dims:] = torch.clamp(var, self.min_var, self.max_var)
        
        phi_project[..., self.num_dims:] += mu2
        return phi_project

    def reparam(self, params_in):
        mu = params_in[..., 0:self.num_dims].clone()
        var = self.min_var + torch.sigmoid(params_in[..., self.num_dims:]) * (self.max_var - self.min_var)
        return torch.cat((mu, var + mu**2), -1)

    def sufficient_statistics(self, x):
        if len(x.shape) == 2:
            stats = torch.stack((x, x ** 2), -1)
        elif len(x.shape) == 3:
            stats = torch.cat((x, x**2), -1)
        else:
            raise AssertionError("Input must be 2 or 3 dimensional tensor.")
        return stats

    def expectation_to_natural(self, phi):
        var = phi[..., self.num_dims:] - phi[..., 0:self.num_dims] ** 2
        var = torch.clamp(var, min=self.min_var, max=self.max_var) # I changed this
        theta1 = phi[..., 0:self.num_dims] / var
        theta2 = - 1. / (2. * var)
        return torch.cat((theta1, theta2), -1)

    def log_normalizer(self, theta):
        log_normalizer = -theta[..., 0:self.num_dims] ** 2 / (4 * theta[..., self.num_dims:]) - 0.5 * torch.log(-2. * theta[..., self.num_dims:])
        log_normalizer = torch.sum(log_normalizer, -1)
        return log_normalizer

    def log_h(self, x):
        return -0.5 * self.log_2pi * self.num_dims

    def _sample(self, num_samples, params, std_correction=1.0):
        with torch.no_grad():
            mu = params[..., 0:self.num_dims]
            var = params[..., self.num_dims:] - mu**2
            std = torch.sqrt(var).unsqueeze(0)
            # print('mu.min, mu.max, mu.mean', mu.min(), mu.max(), mu.mean())

            # exit(mu.shape)
            # from torchvision.utils import save_image
            # mu = mu.reshape(-1, 3, 32, 32)
            # save_image(mu, "mus.png", nrow=6)
            # exit(mu.shape)

            rand = torch.randn((num_samples,) + mu.shape, dtype=mu.dtype, device=mu.device)
            samples = mu.unsqueeze(0) + rand * (std * std_correction)
            #print(samples.shape)
            samples = shift_last_axis_to(samples, 2)
            #exit(samples.shape)
            return samples

    def _argmax(self, params, **kwargs):
        with torch.no_grad():
            mu = params[..., 0:self.num_dims]
            return shift_last_axis_to(mu, 1)

