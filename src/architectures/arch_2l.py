import torch.nn as nn
from src.architectures.layers.SplitActivation import SplitActivation
from src.architectures.layers.ResBlock import ResBlockOord, ResBlock


def _build_arch_2l(conv_params, input_dim, output_dim, latent_dim, dims_e, dims_d, final_act, vae=False):
    e_params, d_params = conv_params
    if len(dims_e) < 1:
        raise ValueError("dims must be a list with <1 element")
    if not(len(e_params) == 2 and len(d_params) == 2):
        raise ValueError("conv_params must be with 2 elements")

    encoder = nn.Sequential(
        nn.Conv2d(input_dim, dims_e[0], **e_params[0]),
        nn.BatchNorm2d(dims_e[0]),
        # nn.LeakyReLU(True),
        nn.ReLU(True),
        
        nn.Conv2d(dims_e[0], latent_dim, **e_params[1]),
        # nn.BatchNorm2d(latent_dim),
        # nn.LeakyReLU(True),
        # nn.ReLU(True),

        # ResBlockOord(latent_dim),
        # ResBlockOord(latent_dim),
        ResBlock(latent_dim),
        ResBlock(latent_dim),
    )

    ld = latent_dim if not vae else latent_dim // 2
    decoder = nn.Sequential(
        # ResBlockOord(ld),
        # ResBlockOord(ld),
        ResBlock(ld),
        ResBlock(ld),

        nn.ConvTranspose2d(ld, dims_d[0], **d_params[0]),
        nn.BatchNorm2d(dims_d[0]),
        # nn.LeakyReLU(True),
        nn.ReLU(True),
        
        nn.ConvTranspose2d(dims_d[0], output_dim, **d_params[1]),
        final_act
    )

    return encoder, decoder

def arch_con_2l(conv_params, input_dim, output_dim, latent_dim, dims_e, dims_d, vae=False, **kwargs):
    final_act = SplitActivation.setup(kwargs["min_max_std"], kwargs["mu_act"], output_dim)
    return _build_arch_2l(conv_params, input_dim, output_dim, latent_dim, dims_e, dims_d, final_act, vae=vae)

def arch_cat_2l(conv_params, input_dim, output_dim, latent_dim, dims):
    final_act = nn.Softmax(dim=1)
    return _build_arch_2l(conv_params, input_dim, output_dim, latent_dim, dims, final_act)
