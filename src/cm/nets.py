import torch.nn as nn
from src.architectures.layers.ResBlock import ResBlockOord

def celeba_conv_decoder_128(
    latent_dim: int,
    n_filters: int,
    batch_norm: bool = True,
    final_act: nn.Module = None,
    bias: bool = False,
    learn_std: bool = False,
    resblock: bool = False,
    out_channels: int = None,
    n_layers: int = 0
):
    """
    Decoder that produces 128x128 images (for CelebA).
    Based on the 64x64 decoder but with one extra upsampling stage:
      1x1 -> 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
    """
    nf = n_filters
    nf2 = max(1, nf // 2)  # intermediate channel reduction
    nf4 = max(1, nf // 4)  # further channel reduction for 128x128

    decoder = nn.Sequential()
    decoder.append(nn.Unflatten(1, (latent_dim, 1, 1)))
    decoder.append(nn.ConvTranspose2d(latent_dim, nf * 4, 4, 1, 0, bias=bias))
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf * 4))
    decoder.append(nn.LeakyReLU())
    if resblock:
        decoder.append(ResBlockOord(nf * 4))

    # state size. (nf*4) x 4 x 4
    decoder.append(nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=bias))
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf * 2))
    decoder.append(nn.LeakyReLU())
    if resblock:
        decoder.append(ResBlockOord(nf * 2))

    # state size. (nf*2) x 8 x 8
    decoder.append(nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1, bias=bias))
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf))
    decoder.append(nn.LeakyReLU())
    if resblock:
        decoder.append(ResBlockOord(nf))

    # state size. (nf) x 16 x 16
    # 16x16 -> 32x32
    decoder.append(nn.ConvTranspose2d(nf, nf2, 4, 2, 1, bias=bias))
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf2))
    decoder.append(nn.LeakyReLU())
    if resblock:
        decoder.append(ResBlockOord(nf2))

    # state size. (nf2) x 32 x 32
    # 32x32 -> 64x64
    decoder.append(nn.ConvTranspose2d(nf2, nf4, 4, 2, 1, bias=bias))
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf4))
    decoder.append(nn.LeakyReLU())
    if resblock:
        decoder.append(ResBlockOord(nf4))

    # state size. (nf4) x 64 x 64
    # final upsample 64x64 -> 128x128
    if not out_channels:
        out_channels = 6 if learn_std else 3
    decoder.append(nn.ConvTranspose2d(nf4, out_channels, 4, 2, 1, bias=bias))

    if final_act is not None:
        decoder.append(final_act)

    return decoder

def celeba_conv_decoder_32(
    latent_dim: int,
    n_filters: int,
    batch_norm: bool = True,
    final_act: nn.Module = None,
    bias: bool = False,
    learn_std: bool = False,
    resblock: bool = False,
    out_channels: int = None,
    n_layers: int = 4
):
    """
    Decoder that produces 32x32 images.
    Based on celeba_conv_decoder but without the final redundant Conv2D layer:
      1x1 -> 4x4 -> 8x8 -> 16x16 -> 32x32
    """
    nf = n_filters

    decoder = nn.Sequential()
    decoder.append(nn.Unflatten(1, (latent_dim, 1, 1)))
    decoder.append(nn.ConvTranspose2d(latent_dim, nf * 4, 4, 1, 0, bias=bias))
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf * 4))
    decoder.append(nn.LeakyReLU())
    if resblock:
        decoder.append(ResBlockOord(nf * 4))

    # state size. (nf*4) x 4 x 4
    decoder.append(nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=bias))
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf * 2))
    decoder.append(nn.LeakyReLU())
    if resblock:
        decoder.append(ResBlockOord(nf * 2))

    # state size. (nf*2) x 8 x 8
    decoder.append(nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1, bias=bias))
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf))
    decoder.append(nn.LeakyReLU())
    if resblock:
        decoder.append(ResBlockOord(nf))

    # state size. (nf) x 16 x 16
    if not out_channels:
        out_channels = 6 if learn_std else 3
    decoder.append(nn.ConvTranspose2d(nf, out_channels, 4, 2, 1, bias=bias))
    # state size. (out_channels) x 32 x 32

    if final_act is not None:
        decoder.append(final_act)

    return decoder

def celeba_conv_decoder_64(
    latent_dim: int,
    n_filters: int,
    batch_norm: bool = True,
    final_act: nn.Module = None,
    bias: bool = False,
    learn_std: bool = False,
    resblock: bool = False,
    out_channels: int = None
):
    """
    Decoder that produces 64x64 images (for CelebA).
    Based on your SVHN decoder but with one extra upsampling stage:
      1x1 -> 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64
    """
    nf = n_filters
    nf2 = max(1, nf // 2)  # intermediate channel reduction; change to `nf` if you prefer

    decoder = nn.Sequential()
    decoder.append(nn.Unflatten(1, (latent_dim, 1, 1)))
    decoder.append(nn.ConvTranspose2d(latent_dim, nf * 4, 4, 1, 0, bias=bias))
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf * 4))
    decoder.append(nn.LeakyReLU())
    if resblock:
        decoder.append(ResBlockOord(nf * 4))

    # state size. (nf*4) x 4 x 4
    decoder.append(nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=bias))
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf * 2))
    decoder.append(nn.LeakyReLU())

    if resblock:
        decoder.append(ResBlockOord(nf * 2))

    # state size. (nf*2) x 8 x 8
    decoder.append(nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1, bias=bias))
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf))
    decoder.append(nn.LeakyReLU())

    if resblock:
        decoder.append(ResBlockOord(nf))

    # state size. (nf) x 16 x 16
    # --- EXTRA UPSAMPLING STAGE FOR 64x64 ---
    # 16x16 -> 32x32
    decoder.append(nn.ConvTranspose2d(nf, nf2, 4, 2, 1, bias=bias))
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf2))
    decoder.append(nn.LeakyReLU())

    if resblock:
        decoder.append(ResBlockOord(nf2))

    # state size. (nf2) x 32 x 32
    # final upsample 32x32 -> 64x64
    if not out_channels:
        out_channels = 6 if learn_std else 3
    decoder.append(nn.ConvTranspose2d(nf2, out_channels, 4, 2, 1, bias=bias))

    if final_act is not None:
        decoder.append(final_act)

    return decoder


def mnist_conv_decoder(
    latent_dim: int, 
    n_filters: int,
    batch_norm: bool = True, 
    final_act: nn.Module = None, 
    bias: bool = False, 
    learn_std: bool = False, 
    resblock: bool = False, 
    out_channels: int = None,
    n_layers: int = 4
):
    dec_net = {
        2: mnist_conv_decoder_2_layers,
        3: mnist_conv_decoder_3_layers,
        4: mnist_conv_decoder_4_layers,
    }
    return dec_net[n_layers](
        latent_dim, 
        n_filters,
        batch_norm, 
        final_act, 
        bias, 
        learn_std, 
        resblock, 
        out_channels,
    )
    

def mnist_conv_decoder_4_layers(
    latent_dim: int, 
    n_filters: int,
    batch_norm: bool = True, 
    final_act: nn.Module = None, 
    bias: bool = False, 
    learn_std: bool = False, 
    resblock: bool = False, 
    out_channels: int = None
):
    nf = n_filters
    decoder = nn.Sequential()   
    
    decoder.append(nn.Unflatten(1, (latent_dim, 1, 1)))
    decoder.append(nn.ConvTranspose2d(latent_dim, nf * 4, 3, 2, 0, bias=bias))
    
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf * 4))
    decoder.append(nn.LeakyReLU())
    if resblock:
        decoder.append(ResBlockOord(nf * 4))

    # state size. (nf*8) x 4 x 4
    decoder.append(nn.ConvTranspose2d(nf * 4, nf * 2, 3, 2, 0, bias=bias))
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf * 2))
    decoder.append(nn.LeakyReLU())
    if resblock:
        decoder.append(ResBlockOord(nf * 2))

    # state size. (nf*4) x 8 x 8
    decoder.append(nn.ConvTranspose2d(nf * 2, nf, 3, 2, 0, bias=bias))
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf))
    decoder.append(nn.LeakyReLU())

    if resblock:
        decoder.append(ResBlockOord(nf))

    # state size. (nf*2) x 16 x 16
    if not out_channels:
        out_channels = 2 if learn_std else 1
    decoder.append(nn.ConvTranspose2d(nf, out_channels, 3, 2, 2, 1, bias=bias))

    if final_act is not None:
        decoder.append(final_act)

    return decoder


def mnist_conv_decoder_3_layers(
    latent_dim: int, 
    n_filters: int,
    batch_norm: bool = True, 
    final_act: nn.Module = None, 
    bias: bool = False, 
    learn_std: bool = False, 
    resblock: bool = False, 
    out_channels: int = None
):
    nf = n_filters
    decoder = nn.Sequential()

    # Start from (B, latent_dim) → (B, latent_dim, 1, 1)
    decoder.append(nn.Unflatten(1, (latent_dim, 1, 1)))

    # ConvT1: 1x1 → 7x7
    decoder.append(nn.ConvTranspose2d(latent_dim, nf * 2, kernel_size=7, stride=1, padding=0, bias=bias))  # → (B, nf*4, 4, 4)
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf * 2))
    decoder.append(nn.LeakyReLU())
    if resblock:
        decoder.append(ResBlockOord(nf * 2))

    # ConvT2: 7x7 → 14x14
    decoder.append(nn.ConvTranspose2d(nf * 2, nf, kernel_size=8, stride=1, padding=0, bias=bias))  # → (B, nf*2, 10, 10)
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf))
    decoder.append(nn.LeakyReLU())
    if resblock:
        decoder.append(ResBlockOord(nf))

    # ConvT3: 14x14 → 28x28
    decoder.append(nn.ConvTranspose2d(nf, out_channels, kernel_size=15, stride=1, padding=0, bias=bias))  # → (B, nf, 20, 20)

    
    if final_act is not None:
        decoder.append(final_act)

    return decoder


def mnist_conv_decoder_2_layers(
    latent_dim: int, 
    n_filters: int,
    batch_norm: bool = True, 
    final_act: nn.Module = None, 
    bias: bool = False, 
    learn_std: bool = False, 
    resblock: bool = False, 
    out_channels: int = None,
):
    nf = n_filters
    decoder = nn.Sequential()

    # Start from (B, latent_dim) → (B, latent_dim, 1, 1)
    decoder.append(nn.Unflatten(1, (latent_dim, 1, 1)))

    # ConvT1: 1x1 → 8x8
    decoder.append(nn.ConvTranspose2d(latent_dim, nf * 2, kernel_size=8, stride=1, padding=0, bias=bias))  # → (B, nf*4, 4, 4)
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf * 2))
    decoder.append(nn.LeakyReLU())
    if resblock:
        decoder.append(ResBlockOord(nf * 2))

    # ConvT2: 8x8 → 28x28
    decoder.append(nn.ConvTranspose2d(nf*2, out_channels, kernel_size=21, stride=1, padding=0, bias=bias))  # → (B, nf, 20, 20)

    
    if final_act is not None:
        decoder.append(final_act)

    return decoder



def svhn_conv_decoder(
    latent_dim: int,
    n_filters: int,
    batch_norm: bool = True,
    final_act: nn.Module = None,
    bias: bool = False,
    learn_std: bool = False,
    resblock: bool = False,
    out_channels: int = None,
    n_layers: int = 4
):
    nf = n_filters
    decoder = nn.Sequential()
    decoder.append(nn.Unflatten(1, (latent_dim, 1, 1)))
    decoder.append(nn.ConvTranspose2d(latent_dim, nf * 4, 4, 1, 0, bias=bias))
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf * 4))
    decoder.append(nn.LeakyReLU())
    if resblock:
        decoder.append(ResBlockOord(nf * 4))

    # state size. (nf*4) x 4 x 4
    decoder.append(nn.ConvTranspose2d(nf * 4, nf * 2, 4, 2, 1, bias=bias))
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf * 2))
    decoder.append(nn.LeakyReLU())

    if resblock:
        decoder.append(ResBlockOord(nf * 2))

    # state size. (nf*2) x 8 x 8
    decoder.append(nn.ConvTranspose2d(nf * 2, nf, 4, 2, 1, bias=bias))
    if batch_norm:
        decoder.append(nn.BatchNorm2d(nf))
    decoder.append(nn.LeakyReLU())

    if resblock:
        decoder.append(ResBlockOord(nf))

    # state size. (nf) x 16 x 16
    if not out_channels:
        out_channels = 6 if learn_std else 3
    decoder.append(nn.ConvTranspose2d(nf, out_channels, 4, 2, 1, bias=bias))

    if final_act is not None:
        decoder.append(final_act)
    

    return decoder