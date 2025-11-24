from src.utilities import seed_everything
import json
import argparse
from src.scripts.eval.eval_vqvae import eval_vqvae
from src.scripts.eval.eval_pcnn import eval_pcnn
from src.scripts.eval.eval_cm import eval_cm
from src.scripts.eval.eval_einet import eval_einet
from src.scripts.eval.eval_dm import eval_dm

def arguments():
    parser = argparse.ArgumentParser(description='Arguments')


    parser.add_argument(
        '--config',
        '-c',
        help='Define the path to the config file.'
    )

    parser.add_argument(
        '--results_dir',
        '-rd',
        help='Define the path to the results directory.',
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

CFS = {
    "vqvae": ("./config/eval/vqvae.json", eval_vqvae),
    "pcnn": ("./config/eval/pcnn.json", eval_pcnn),
    "dm": ("./config/eval/dm.json", eval_dm),
    "cm": ("./config/eval/cm.json", eval_cm),
    "einet": ("./config/eval/einet.json", eval_einet),
}

if __name__ == '__main__':
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["WANDB_MODE"] = "disabled"
    args = arguments()
    
    if args.model_type not in CFS:
        raise ValueError(f"Unsupported model type: '{args.model_type}'.")
    config_file, eval_fun = CFS[args.model_type]    
    if args.config is not None:
        config_file = args.config

    config=None
    with open(config_file, 'r') as json_file:
        config = json.load(json_file)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = args.models_dir
    if models_dir is None:
        models_dir = f"{current_dir}/models"

    results_dir = args.results_dir
    if results_dir is None:
        results_dir = f"{current_dir}/results"
    
    seed_everything(seed=config["seed"])
    eval_fun(
        config,
        models_dir,
        results_dir
    )
