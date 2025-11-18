import torch
import numpy as np
import json
from src.utilities import latent_dim, seed_everything, set_determinism
from src.scripts.model_loaders import load_pcnn, load_psnail, load_vqvae, load_vae, load_ae
from datasets.data import data_loaders
from src.searches.beam_search import beam_search_random_start_c
import argparse
import os

from src.scripts.model_savers import save_discrete_latents, save_continuous_latents
from src.scripts.model_loaders import load_dm

SAVE = {
    "pcnn": save_discrete_latents,
    "psnail": save_discrete_latents,
    "vqvae": save_discrete_latents,

    "vae": save_continuous_latents,

    "ae": save_continuous_latents,
}

def rs_distillation(config, models_dir, prior):
    n = config["n_components"]
    seeds = config.get("seeds", [0])
    print(seeds)
    seed_everything(seed=config["seed"])
    latents, _ = prior._params(n, unique=True, batch_size=2**10)
    SAVE[config["prior"]](models_dir, config, latents, seed=0)
    return latents


def bs_distillation(config, models_dir, prior):
    n = config["n_components"]
    # latents, _ = beam_search_cs_unique(prior, B=n)
    latents, _ = beam_search_random_start_c(prior, n, S=2**4, verbose=True)
    print(n)
    print(latents.unique(dim=0).shape)
    SAVE[config["prior"]](models_dir, config, latents, seed = 0)

DISTILLATION_TYPES = {
    "RS": rs_distillation,
    "BS": bs_distillation,
}

def distillation(config, models_dir):
    load_prior_model = {
        "pcnn": load_pcnn,
    }

    print(config)
    prior, _ = load_prior_model[config["prior"]](models_dir, config)
    latents = DISTILLATION_TYPES[config["distillation_type"]](config, models_dir, prior)



def arguments():
    parser = argparse.ArgumentParser(description='Arguments')

    parser.add_argument(
        '--config',
        '-c',
        help='Define the path to the config file.',
        default=None
    )


    return parser.parse_args()

if __name__ == '__main__':
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    args = arguments()

    config_file = f"./config/distillation.json"
    if args.config is not None:
        config_file = args.config

    current_dir = os.path.dirname(os.path.abspath(__file__))

    models_dir = f"{current_dir}/experimental_settings/models"   
    
    config=None
    with open(config_file, 'r') as json_file:
        config = json.load(json_file)

    rdir = f"{current_dir}/grid_search_evals13"
    
    distillation(config, models_dir)
    dm, model_name = load_dm(models_dir, config, config["seed"])


    data_cnf = config["input_data"]

    pixel = data_cnf["pixel_representation"]["type"]

    rdir = f'{rdir}/{data_cnf["dataset"]}/{pixel}/dm_{model_name}'
    print(rdir)
    os.makedirs(rdir, exist_ok=True)

    from src.scripts.eval.eval_dm import performance

    seed_everything(seed=config["seed"])
    performance(models_dir, config, rdir, [0], mode="test")
