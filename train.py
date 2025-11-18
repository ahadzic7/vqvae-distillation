import json
import argparse
from src.scripts.train.train_vqvae import train_vqvae
from src.scripts.train.train_vqvae_rec import train_vqvae_rec

from src.scripts.train.train_pcnn import train_pcnn
from src.scripts.train.train_vqvae_pcnn import train_vqvae_pcnn
from src.scripts.train.train_cm import train_cm
from src.scripts.train.train_classifier import train_classifier
from src.scripts.train.train_einet import train_einet
from src.scripts.train.train_psnail import train_psnail
from src.scripts.train.train_vqvae_pcnn_dm import train_vqvae_pcnn_dm
from src.scripts.train.train_vae import train_vae
from src.scripts.train.train_ae import train_ae
from src.scripts.train.train_vae_dm import train_vae_dm, train_ae_dm


# from scripts.train.train_mcvqvae import train_mcvqvae
# from scripts.train.train_mcpcnn import train_mcpcnn

def arguments():
    parser = argparse.ArgumentParser(description='Arguments')

    parser.add_argument(
        '--config',
        '-c',
        help='Define the path to the config file.',
        default=None
    )

    parser.add_argument(
        '--model_type',
        '-mt',
        help='Define the model to use.',
        default=None
    )

    parser.add_argument(
        '--models_dir',
        '-md',
        help='Define the path to the models directory.',
        default=None
    )

    return parser.parse_args()


if __name__ == '__main__':
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    args = arguments()

    cfs = {
        "ae": ("./config/train/ae.json", train_ae),
        "vqvae-rec": ("./config/train/vqvae_rec.json", train_vqvae_rec),

        "vae": ("./config/train/vae.json", train_vae),
        "vqvae": ("./config/train/vqvae.json", train_vqvae),
        "pcnn": ("./config/train/pcnn.json", train_pcnn),
        
        "vqvae-pcnn": ("./config/train/vqvae-pcnn.json", train_vqvae_pcnn),
        "vqvae-pcnn-dm": ("./config/train/vqvae-pcnn-dm.json", train_vqvae_pcnn_dm),
        "vae-dm": ("./config/train/vae-dm.json", train_vae_dm),
        "ae-dm": ("./config/train/ae-dm.json", train_ae_dm),

        "cm": ("./config/train/cm.json", train_cm),

        "classifier": ("./config/train/classifier.json", train_classifier),
        "einet": ("./config/train/einet.json", train_einet),
        "psnail": ("./config/train/psnail.json", train_psnail),

        # "mcvqvae": ("./config/train/mcvqvae.json", train_mcvqvae),
        # "mcpcnn": ("./config/train/mcpcnn.json", train_mcpcnn),
    }
    
    if args.model_type not in cfs:
        raise ValueError(f"Unsupported model type: '{args.model_type}'.")
    config_file, train_fun = cfs[args.model_type]    
    if args.config is not None:
        config_file = args.config

    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = args.models_dir
    if models_dir is None:
        # models_dir = f"{current_dir}/models"
        # models_dir = f"{current_dir}/grid_search"
        models_dir = f"{current_dir}/debug2"
    
    config=None
    with open(config_file, 'r') as json_file:
        config = json.load(json_file)
    
    from utilities import seed_everything
    seed_everything(seed=config["seed"])
    train_fun(config, models_dir)
    