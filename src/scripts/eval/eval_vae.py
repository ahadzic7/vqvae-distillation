import torch
from src.utilities import *
from src.mm.GaussianMixture import GaussianMixture as GMM

from src.metrics import mse_loss, nll_loss, fid_comparison
from src.scripts.model_loaders import load_vae
from src.scripts.eval.eval_vqvae import images, reconstructions
from datasets.data import data_loaders
import os
import numpy as np

def eval_vae(
        config, 
        models_dir=None,
        rdir=None,
    ):
    data = config["input_data"]
    
    vae, model_name = load_vae(models_dir, config)
    pixel = data["pixel_representation"]["type"]
    rdir = f'{rdir}/{data["dataset"]}/{pixel}/dm_{model_name}'
    print(rdir)
    os.makedirs(rdir, exist_ok=True) 
    
    _, _, testl = data_loaders(config["input_data"])
    if data["supervised"]["use_labeling"]:
        reconstructions(vae, config, rdir)
    else:
        images(vae, config, rdir, testl)

    # n = 10**4
    # print(n)
    # fid = fid_comparison(vae, testl, n, vae.device)
    # print(f"FID: {fid:.2f}")


    mse = mse_loss(vae, testl)
    print(f"MSE: {mse:.2f}")
    nll = nll_loss(vae, testl)
    print(f"NLL: {nll:.2f}")

    M = data["input_shape"]["height"] * data["input_shape"]["width"] * data["input_shape"]["channels"]
    Dlog2 = M * np.log(2)
    bpd = nll / Dlog2 + 8
    print(f"BPD: {bpd:.2f}")
