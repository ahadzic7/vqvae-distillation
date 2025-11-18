import torch.nn as nn
from src.architectures.layers.SplitActivation import SplitActivation
from src.architectures.layers.ResBlock import ResBlockOord, ResBlock


def _build_arch_1l(conv_params, input_dim, output_dim, latent_dim, dims_e, dims_d, final_act, vae=False):
    e_params, d_params = conv_params
    if not(len(e_params) == 1 and len(d_params) == 1):
        raise ValueError("conv_params must be with 1 elements")

    encoder = nn.Sequential(
        nn.Conv2d(input_dim, latent_dim, **e_params[0]),
        nn.BatchNorm2d(latent_dim),
        nn.ReLU(),
        # nn.LeakyReLU(True),

        ResBlock(latent_dim),
        ResBlock(latent_dim),
        # ResBlockOord(latent_dim),
        # ResBlockOord(latent_dim),
    )

    ld = latent_dim if not vae else latent_dim // 2
    decoder = nn.Sequential(
        # ResBlockOord(ld),
        # ResBlockOord(ld),
        ResBlock(ld),
        ResBlock(ld),
        
        nn.ConvTranspose2d(ld, output_dim, **d_params[0]),
        # nn.BatchNorm2d(output_dim),
        # nn.LeakyReLU(True),

        final_act
    )

    return encoder, decoder

def arch_con_1l(conv_params, input_dim, output_dim, latent_dim, dims_e, dims_d, vae=False, **kwargs):
    final_act = SplitActivation.setup(kwargs["min_max_std"], kwargs["mu_act"], output_dim)
    return _build_arch_1l(conv_params, input_dim, output_dim, latent_dim, dims_e, dims_d, final_act, vae=vae)

def arch_cat_1l(conv_params, input_dim, output_dim, latent_dim, dims):
    final_act = nn.Softmax(dim=1)
    return _build_arch_1l(conv_params, input_dim, output_dim, latent_dim, dims, final_act)
