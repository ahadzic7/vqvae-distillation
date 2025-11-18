from src.scripts.train.train_vae import train_vae
from src.scripts.train.train_ae import train_ae
from distillation import DISTILLATION_TYPES

def train_vae_dm(config, models_dir=None, vae=None):
    vae = train_vae(config, models_dir).to("cuda")

    dt = config["distillation_type"]

    DISTILLATION_TYPES[dt](config, models_dir, vae)
    
    
def train_ae_dm(config, models_dir=None, vae=None):
    ae = train_ae(config, models_dir).to("cuda")

    dt = config["distillation_type"]

    DISTILLATION_TYPES[dt](config, models_dir, ae)    