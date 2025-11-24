import json
import argparse
from src.scripts.train.train_vqvae_pcnn_dm import train_vqvae_pcnn_dm
from src.scripts.train.train_cm import train_cm
from src.scripts.train.train_einet import train_einet
from src.utilities import seed_everything



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

    return parser.parse_args()


if __name__ == '__main__':
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    args = arguments()

    cfs = {
        "./experimental_settings/plot/dm-pipeline-RS-1.json": train_vqvae_pcnn_dm,
        "./experimental_settings/plot/dm-pipeline-RS-2.json": train_vqvae_pcnn_dm,
        "./experimental_settings/plot/dm-pipeline-BS.json": train_vqvae_pcnn_dm,

        "./experimental_settings/plot/cm-pipeline.json": train_cm,
        "./experimental_settings/plot/einet-pipeline.json": train_einet,
        
        "./experimental_settings/plot/exact_models/dm-pipeline-1.json": train_vqvae_pcnn_dm,
        "./experimental_settings/plot/exact_models/dm-pipeline-2.json": train_vqvae_pcnn_dm,
        "./experimental_settings/plot/exact_models/dm-pipeline-3.json": train_vqvae_pcnn_dm,
        "./experimental_settings/plot/exact_models/dm-pipeline-4.json": train_vqvae_pcnn_dm,
        "./experimental_settings/plot/exact_models/dm-pipeline-5.json": train_vqvae_pcnn_dm,
        "./experimental_settings/plot/exact_models/dm-pipeline-6.json": train_vqvae_pcnn_dm,
        
        "./experimental_settings/inpaints/dm-pipeline-RS.json": train_vqvae_pcnn_dm,
        "./experimental_settings/inpaints/dm-pipeline-BS.json": train_vqvae_pcnn_dm,

    }
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = f"{current_dir}/models"
    rdir = f"{current_dir}/results"


    for config_file in cfs.keys():
        train_fun = cfs[config_file]
        with open(config_file, 'r') as json_file:
            config = json.load(json_file)
        
        seed_everything(seed=config["seed"])
        train_fun(config, models_dir=models_dir, rdir=rdir)
 