import torch.nn as nn
from src.architectures.layers.SplitActivation import SplitActivation
from src.architectures.layers.ResBlock import ResBlockOord, ResBlock

def _build_arch_5l(conv_params, input_dim, output_dim, latent_dim, dims_e, dims_d, final_act, vae=False):
    e_params, d_params = conv_params
    if len(dims_e) < 1:
        raise ValueError("dims must be a list with <1 element")
    if not(len(e_params) == 5 and len(d_params) == 5):
        raise ValueError("conv_params must be with 4 elements")
    
    encoder = nn.Sequential(
        nn.Conv2d(input_dim, dims_e[0], **e_params[0]),
        nn.BatchNorm2d(dims_e[0]),
        # nn.LeakyReLU(), 
        nn.ReLU(True),

        nn.Conv2d(dims_e[0], dims_e[1], **e_params[1]),
        nn.BatchNorm2d(dims_e[1]),
        # nn.LeakyReLU(), 
        nn.ReLU(True),

        nn.Conv2d(dims_e[1], dims_e[2], **e_params[2]),
        nn.BatchNorm2d(dims_e[2]),
        # nn.LeakyReLU(), 
        nn.ReLU(True),

        nn.Conv2d(dims_e[2], dims_e[3], **e_params[3]),
        nn.BatchNorm2d(dims_e[3]),
        # nn.LeakyReLU(), 
        nn.ReLU(True),

        nn.Conv2d(dims_e[3], latent_dim, **e_params[4]),
        nn.BatchNorm2d(latent_dim),
        # nn.LeakyReLU(), 
        nn.ReLU(True),

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

        nn.ConvTranspose2d(ld, dims_d[3], **d_params[0]),
        nn.BatchNorm2d(dims_d[3]),
        nn.ReLU(True),

        nn.ConvTranspose2d(dims_d[3], dims_d[2], **d_params[1]),
        nn.BatchNorm2d(dims_d[2]),
        nn.ReLU(True),

        nn.ConvTranspose2d(dims_d[2], dims_d[1], **d_params[2]),
        nn.BatchNorm2d(dims_d[1]),
        nn.ReLU(True),

        nn.ConvTranspose2d(dims_d[1], dims_d[0], **d_params[3]),
        nn.BatchNorm2d(dims_d[0]),
        nn.ReLU(True),

        nn.ConvTranspose2d(dims_d[0], output_dim, **d_params[4]),
        final_act
    )
    return encoder, decoder

def arch_con_5l(conv_params, input_dim, output_dim, latent_dim, dims_e, dims_d, vae=False, **kwargs):
    final_act = SplitActivation.setup(kwargs["min_max_std"], kwargs["mu_act"], output_dim)
    return _build_arch_5l(conv_params, input_dim, output_dim, latent_dim, dims_e, dims_d, final_act, vae=vae)

def arch_cat_5l(conv_params, input_dim, output_dim, latent_dim, dims):
    final_act = nn.Softmax(dim=1)
    return _build_arch_5l(conv_params, input_dim, output_dim, latent_dim, dims, final_act)
    e_params, d_params = conv_params
    if len(dims) < 1:
        raise ValueError("dims must be a list with <1 element")
    if not(len(e_params) == 4 and len(d_params) == 4):
        raise ValueError("conv_params must be with 4 elements")
    final_act = SplitActivation.setup(kwargs["min_max_std"], kwargs["mu_act"], output_dim)

    encoder = nn.Sequential(
        nn.Conv2d(input_dim, dims[0], **e_params[0]), # 421
        nn.LeakyReLU(),
        nn.Conv2d(dims[0],dims[1],**e_params[1]), #421
        nn.LeakyReLU(),
        nn.Conv2d(dims[1],dims[2],**e_params[2]),#311
        nn.LeakyReLU(),
        ResBlockOord(dims[2]),
        ResBlockOord(dims[2]),
        ResBlockOord(dims[2]),
        ResBlockOord(dims[2]),
        ResBlockOord(dims[2]),
        ResBlockOord(dims[2]),
        nn.LeakyReLU(),
        nn.Conv2d(dims[2], latent_dim, **e_params[3]),#11
        nn.LeakyReLU(),
    )

    decoder = nn.Sequential(
        nn.Conv2d(latent_dim, dims[2],**d_params[1]),
        nn.LeakyReLU(),
        ResBlockOord(dims[2]),
        ResBlockOord(dims[2]),
        ResBlockOord(dims[2]),
        ResBlockOord(dims[2]),
        ResBlockOord(dims[2]),
        ResBlockOord(dims[2]),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(dims[2],dims[0],**d_params[2]),
        nn.LeakyReLU(),
        nn.ConvTranspose2d(dims[0],output_dim,**d_params[3]),
        final_act
    )

    return encoder, decoder