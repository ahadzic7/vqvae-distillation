import torch.nn as nn

def vqvae_cat(
        input_dim=256,
        output_dim=256,
        latent_dim=16,
        codebook_size=512, 
        dims=[32, 64, 96]
    ):

    decoder = nn.Sequential(
        nn.ConvTranspose2d(latent_dim, dims[2], 3, 1, 0),
        nn.BatchNorm2d(dims[2]),
        nn.LeakyReLU(True),

        nn.ConvTranspose2d(dims[2], dims[1], 5, 1, 0),
        nn.BatchNorm2d(dims[1]),
        nn.LeakyReLU(True),

        nn.ConvTranspose2d(dims[1], dims[0], 4, 2, 1),
        nn.BatchNorm2d(dims[0]),
        nn.LeakyReLU(True),

        nn.ConvTranspose2d(dims[0], output_dim, 4, 2, 1),
        nn.Softmax(dim=1)
    )
    return decoder
