import torch.nn as nn
import torch
import torch.nn.functional as F
from src.SplitActivation import SplitActivation

# from src.ResBlock import ResBlockOord

def setup(min_max_std, mu_act: str, output_dim, *, device=None):
    if min_max_std is None and output_dim != 2:
        raise AttributeError("When learning σ you must output μ|σ (channels=2)")
 
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    min_std, max_std = min_max_std
    min_log_var = 2 * torch.log(torch.tensor(min_std, device=device))
    max_log_var = 2 * torch.log(torch.tensor(max_std, device=device))
 
    mu_activations = {"tanh": F.tanh, "sigmoid": F.sigmoid}

    return SplitActivation(
        fc1= mu_activations[mu_act],
        min_max_clamp=(min_log_var, max_log_var),
    )




def vae(
        input_dim,
        output_dim,
        latent_dim,
        dims,
        final_act
    ):
    
    encoder = nn.Sequential(
        nn.Conv2d(input_dim, dims[0], 4, 2, 1),
        nn.BatchNorm2d(dims[0]),
        nn.ReLU(True),

        nn.Conv2d(dims[0], dims[1], 4, 2, 1),
        nn.BatchNorm2d(dims[1]),
        nn.ReLU(True),

        nn.Conv2d(dims[1], dims[2], 5, 1, 0),
        nn.BatchNorm2d(dims[2]),
        nn.ReLU(True),

        nn.Conv2d(dims[2], 2 * latent_dim, 3, 1, 0),
        nn.BatchNorm2d(2 * latent_dim),
    )

    decoder = nn.Sequential(
        nn.ConvTranspose2d(latent_dim, dims[2], 3, 1, 0),
        nn.BatchNorm2d(dims[2]),
        nn.ReLU(True),

        nn.ConvTranspose2d(dims[2], dims[1], 5, 1, 0),
        nn.BatchNorm2d(dims[1]),
        nn.ReLU(True),

        nn.ConvTranspose2d(dims[1], dims[0], 4, 2, 1),
        nn.BatchNorm2d(dims[0]),
        nn.ReLU(True),

        nn.ConvTranspose2d(dims[0], output_dim, 4, 2, 1),
        final_act
    )
    return encoder, decoder



def vae_con(
        input_dim=1,
        output_dim=2,
        latent_dim=128,
        dims=[32, 64, 96],

        mu_act=F.tanh,
        min_max_std=(1e-3, 1.0)
    ):
    mu_act = setup(min_max_std, mu_act, output_dim)
    e, d = vae(input_dim, output_dim, latent_dim, dims, final_act=mu_act)
    return e, d, (latent_dim, 1, 1)




def vae_2_2(
        input_dim,
        output_dim,
        latent_dim,
        dims,
        final_act
    ):
    
    encoder = nn.Sequential(
        nn.Conv2d(input_dim, dims[0], kernel_size=4, stride=2, padding=1),  # 28 → 14
        nn.BatchNorm2d(dims[0]),
        nn.ReLU(True),

        nn.Conv2d(dims[0], dims[1], kernel_size=4, stride=2, padding=1),    # 14 → 7
        nn.BatchNorm2d(dims[1]),
        nn.ReLU(True),

        nn.Conv2d(dims[1], dims[2], kernel_size=4, stride=2, padding=1),    # 7 → 4
        nn.BatchNorm2d(dims[2]),
        nn.ReLU(True),

        nn.Conv2d(dims[2], 2 * latent_dim, kernel_size=3, stride=2, padding=1),  # 4 → 2
        nn.BatchNorm2d(2 * latent_dim),
    )

    decoder = nn.Sequential(
        nn.ConvTranspose2d(latent_dim, dims[2], kernel_size=3, stride=1, padding=0),  # 2 -> 4
        nn.BatchNorm2d(dims[2]),
        nn.ReLU(True),

        nn.ConvTranspose2d(dims[2], dims[1], kernel_size=4, stride=1, padding=0),  # 4 -> 7
        nn.BatchNorm2d(dims[1]),
        nn.ReLU(True),

        nn.ConvTranspose2d(dims[1], dims[0], kernel_size=4, stride=2, padding=1),  # 7 -> 14
        nn.BatchNorm2d(dims[0]),
        nn.ReLU(True),

        nn.ConvTranspose2d(dims[0], output_dim, kernel_size=4, stride=2, padding=1),  # 14 -> 28
        final_act  # or nn.Sigmoid(), depending on your data range
    )
    return encoder, decoder


def vae_con_2_2(
        input_dim=1,
        output_dim=2,
        latent_dim=128,
        dims=[32, 64, 96],

        mu_act=F.tanh,
        min_max_std=(1e-3, 1.0)
    ):
    mu_act = setup(min_max_std, mu_act, output_dim)
    e, d = vae_2_2(input_dim, output_dim, latent_dim, dims, final_act=mu_act)
    return e, d, (latent_dim, 2, 2)



def vae_4_4(
        input_dim,
        output_dim,
        latent_dim,
        dims,
        final_act
    ):
    
    encoder = nn.Sequential(
        nn.Conv2d(input_dim, dims[0], kernel_size=4, stride=2, padding=1),  # 28 → 14
        nn.BatchNorm2d(dims[0]),
        nn.ReLU(True),

        nn.Conv2d(dims[0], dims[1], kernel_size=4, stride=2, padding=1),    # 14 → 7
        nn.BatchNorm2d(dims[1]),
        nn.ReLU(True),

        nn.Conv2d(dims[1], 2*latent_dim, kernel_size=3, stride=2, padding=1),  # 7 → 4
        nn.BatchNorm2d(2 * latent_dim),
        nn.ReLU(True)
    )

    decoder = nn.Sequential(
        # 4×4 → 7×7
        nn.ConvTranspose2d(latent_dim, dims[1],
                        kernel_size=3, stride=2,
                        padding=1, output_padding=0),  # ← change here
        nn.BatchNorm2d(dims[1]),
        nn.ReLU(True),

        # 7×7 → 14×14
        nn.ConvTranspose2d(dims[1], dims[0],
                        kernel_size=4, stride=2,
                        padding=1),
        nn.BatchNorm2d(dims[0]),
        nn.ReLU(True),

        # 14×14 → 28×28
        nn.ConvTranspose2d(dims[0], output_dim,
                        kernel_size=4, stride=2,
                        padding=1),
        final_act
    )
    print("aaa")
    return encoder, decoder


def vae_con_4_4(
        input_dim=1,
        output_dim=2,
        latent_dim=128,
        dims=[32, 64, 96],

        mu_act=F.tanh,
        min_max_std=(1e-3, 1.0)
    ):
    mu_act = setup(min_max_std, mu_act, output_dim)
    e, d = vae_4_4(input_dim, output_dim, latent_dim, dims, final_act=mu_act)
    return e, d, (latent_dim, 4, 4)




def arch_selector(architecture):
    selector = {
        "placeholder_1": vae_con,
        "placeholder2_1": vae_con,
        "placeholder3_1": vae_con,
        "2_2_2": vae_con_2_2,
        "4_4_4": vae_con_4_4,
    }
    return selector[architecture]



# ###
# def vae_cm(
#         input_dim,
#         output_dim,
#         latent_dim,
#         dims,
#         negative_slope,
#         final_act,
#     ):
#     encoder = nn.Sequential(
#         nn.Conv2d(input_dim, dims[0], kernel_size=3, stride=2, bias=False),
#         nn.BatchNorm2d(dims[0]),
#         nn.LeakyReLU(negative_slope=negative_slope),
#         ResBlockOord(dims[0]),

#         nn.Conv2d(dims[0], dims[1], kernel_size=3, stride=2, bias=False),
#         nn.BatchNorm2d(dims[1]),
#         nn.LeakyReLU(negative_slope=negative_slope),
#         ResBlockOord(dims[1]),
        
#         nn.Conv2d(dims[1], latent_dim, kernel_size=3, stride=2, bias=False),
#         nn.BatchNorm2d(latent_dim),
#         nn.LeakyReLU(negative_slope=negative_slope),
#         ResBlockOord(latent_dim),
        
#         nn.Conv2d(latent_dim, 2*latent_dim, kernel_size=2, stride=2, padding=0, bias=False),
#         nn.BatchNorm2d(2*latent_dim),
#     )
    
#     decoder = nn.Sequential(
#         nn.ConvTranspose2d(latent_dim, latent_dim, kernel_size=3, stride=2, bias=False),
#         nn.BatchNorm2d(latent_dim),
#         nn.LeakyReLU(negative_slope=negative_slope),
#         ResBlockOord(latent_dim),

#         nn.ConvTranspose2d(latent_dim, dims[1], kernel_size=3, stride=2, bias=False),
#         nn.BatchNorm2d(dims[1]),
#         nn.LeakyReLU(negative_slope=negative_slope),
#         ResBlockOord(dims[1]),
        
#         nn.ConvTranspose2d(dims[1], dims[0], kernel_size=3, stride=2, bias=False),
#         nn.BatchNorm2d(dims[0]),
#         nn.LeakyReLU(negative_slope=negative_slope),
#         ResBlockOord(dims[0]),
        
#         nn.ConvTranspose2d(dims[0], output_dim, kernel_size=3, stride=2, padding=2, output_padding=1, bias=False),
#     )
#     if final_act is not None:
#         decoder.append(final_act)
    
#     return encoder, decoder, (latent_dim, 1, 1)

# def vae_cat_cm(
#         input_dim=256,
#         output_dim=256,
#         latent_dim=16,
#         dims=[16, 32],
#         negative_slope=0.01
#     ):
#     return vae_cm(
#         input_dim=input_dim,
#         output_dim=output_dim,
#         latent_dim=latent_dim,
#         dims=dims,
#         negative_slope=negative_slope,
#         final_act=nn.Softmax(dim=1)
#     )

# def vae_con_tanh(
#         input_dim=1,
#         output_dim=2,
#         latent_dim=16,
#         dims=[16, 32],
#         negative_slope=0.01
#     ):
#     return vae_cm(
#         input_dim=input_dim,
#         output_dim=output_dim,
#         latent_dim=latent_dim,
#         dims=dims,
#         negative_slope=negative_slope,
#         final_act=nn.Tanh()
#     )

# def vae_con_sigmoid(
#         input_dim=1,
#         output_dim=2,
#         latent_dim=16,
#         dims=[16, 32],
#         negative_slope=0.01
#     ):
#     return vae_cm(
#         input_dim=input_dim,
#         output_dim=output_dim,
#         latent_dim=latent_dim,
#         dims=dims,
#         negative_slope=negative_slope,
#         final_act=nn.Sigmoid()
#     )
# ###

# def vae_cat(
#         input_dim=256,
#         output_dim=256,
#         latent_dim=128,
#         dims=[32, 64, 96]
#     ):

#     encoder = nn.Sequential(
#         nn.Conv2d(input_dim, dims[0], 4, 2, 1),
#         nn.BatchNorm2d(dims[0]),
#         nn.ReLU(True),

#         nn.Conv2d(dims[0], dims[1], 4, 2, 1),
#         nn.BatchNorm2d(dims[1]),
#         nn.ReLU(True),

#         nn.Conv2d(dims[1], dims[2], 5, 1, 0),
#         nn.BatchNorm2d(dims[2]),
#         nn.ReLU(True),

#         nn.Conv2d(dims[2], 2*latent_dim, 3, 1, 0),
#         nn.BatchNorm2d(2*latent_dim),
#     )

#     decoder = nn.Sequential(
#         nn.ConvTranspose2d(latent_dim, dims[2], 3, 1, 0),
#         nn.BatchNorm2d(dims[2]),
#         nn.ReLU(True),

#         nn.ConvTranspose2d(dims[2], dims[1], 5, 1, 0),
#         nn.BatchNorm2d(dims[1]),
#         nn.ReLU(True),

#         nn.ConvTranspose2d(dims[1], dims[0], 4, 2, 1),
#         nn.BatchNorm2d(dims[0]),
#         nn.ReLU(True),

#         nn.ConvTranspose2d(dims[0], output_dim, 4, 2, 1),
#         nn.Softmax(dim=1)
#     )
    
#     return encoder, decoder, (latent_dim, 1, 1)
 
