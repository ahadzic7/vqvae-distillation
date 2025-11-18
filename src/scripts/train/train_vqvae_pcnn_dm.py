from src.scripts.train.train_pcnn import train_pcnn
from src.scripts.train.train_vqvae import train_vqvae
from distillation import DISTILLATION_TYPES
from src.scripts.eval.eval_dm import performance
from src.scripts.model_loaders import ModelLoader
import os
from src.utilities import seed_everything, get_device

def train_vqvae_pcnn_dm(config, models_dir=None, vqvae=None, rdir=None):
    device = get_device()
    model_loader = ModelLoader(models_dir, config)
    vqvae_config = config.get("model", config.get("vqvae", None))
    dt = config["distillation_type"]
    prior = config["prior"]

    arch, layers, ld, cb, beta, seed = model_loader._format_params(vqvae_config)
    model_name = model_loader.path_builder.discrete_dm_name(arch, layers, ld, cb, beta, seed, dt, prior)
    
    data = config["input_data"]
    pixel = data["pixel_representation"]["type"]
    rdir = f'{rdir}/{data["dataset"]}/{pixel}/dm_{model_name}'
    os.makedirs(rdir, exist_ok=True)


    c = config["input_data"]["supervised"]["use_labeling"]
    if c:
        config["input_data"]["supervised"]["use_labeling"] = False

    seed_everything(seed=config["seed"])
    vqvae = train_vqvae(config, models_dir).to(device)

    config["input_data"]["supervised"]["use_labeling"] = c

    seed_everything(seed=config["seed"])

    vqvae = vqvae.eval()
    if dt == "RS":
        pcnn = train_pcnn(config, models_dir, vqvae).to("cuda")
        prior_model = pcnn
        other_dt = "BS"
    elif dt == "BS":
        pcnn = train_pcnn(config, models_dir, vqvae).to("cuda")
        prior_model = pcnn
        other_dt = "RS"
    elif dt == "ML" or dt == "MF":
        prior_model = vqvae
    else:
        raise ValueError(f"Distillation type {dt} not supported for VQ-VAE + PCNN model.")

    seed_everything(seed=config["seed"])
    DISTILLATION_TYPES[dt](config, models_dir, prior_model)
    
    seed_everything(seed=config["seed"])
    performance(models_dir, config, rdir, [0], mode="test")



    # model_name = model_loader.path_builder.discrete_dm_name(arch, layers, ld, cb, beta, seed, other_dt, prior)
    
    # data = config["input_data"]
    # pixel = data["pixel_representation"]["type"]
    # rdir = f'{rdir}/{data["dataset"]}/{pixel}/dm_{model_name}'
    # os.makedirs(rdir, exist_ok=True)




    # DISTILLATION_TYPES[other_dt](config, models_dir, prior_model)
    
    # performance(models_dir, config, rdir, [0], mode="test")
    
    
    