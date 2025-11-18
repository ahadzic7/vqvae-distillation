from src.vae.VAE_con import VAE_con
# from src.vae.VAE_cat import VAE_cat
from src.utilities import *
from src.scripts.train.setup.setup import vae_trainer
from datasets.data import data_loaders
from src.scripts.model_loaders import ModelLoader
from src.scripts.model_savers import save_vae

def train_vae(config, models_dir):
    device = get_device()
    t = VAE_con # if pixel_rep == "con" else VAE_cat 
       
    trainl, validl, _ = data_loaders(config["input_data"])
    
    vae = t(config).to(device)
    
    x = next(iter(trainl))[:32].to(device)
    print(vae.encode(x).shape)
    print(vae.forward(x).shape)

    # Clean and simple!
    loader = ModelLoader(models_dir, config)
    log_folder, name, version = loader.get_training_paths("vae")

    trainer, _, _ = vae_trainer(log_folder, config["epochs"], name=name, version=version)
    trainer.fit(vae, trainl, validl)
    save_vae(models_dir, config, vae)

    return vae
