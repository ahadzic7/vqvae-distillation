import json
import argparse
from src.scripts.train.train_vqvae import train_vqvae
from src.scripts.train.train_pcnn import train_pcnn
from src.scripts.train.train_cm import train_cm
from src.scripts.train.train_einet import train_einet
from src.scripts.train.train_vqvae_pcnn_dm import train_vqvae_pcnn_dm
from utilities import seed_everything


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
        "vqvae": ("./config/train/vqvae.json", train_vqvae),
        "pcnn": ("./config/train/pcnn.json", train_pcnn),

        "vqvae-pcnn-dm": ("./config/train/vqvae-pcnn-dm.json", train_vqvae_pcnn_dm),

        "cm": ("./config/train/cm.json", train_cm),

        "einet": ("./config/train/einet.json", train_einet),
    }
    
    if args.model_type not in cfs:
        raise ValueError(f"Unsupported model type: '{args.model_type}'.")
    config_file, train_fun = cfs[args.model_type]    
    if args.config is not None:
        config_file = args.config

    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = args.models_dir
    if models_dir is None:
        models_dir = f"{current_dir}/models"

    config=None
    with open(config_file, 'r') as json_file:
        config = json.load(json_file)
    
    
    seed_everything(seed=config["seed"])
    train_fun(config, models_dir)
    