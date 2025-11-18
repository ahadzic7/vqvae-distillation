import torch
import json
from datasets.data import data_loaders
from src.scripts.model_loaders import load_pcnn
from src.metrics import bpd_formula
from tqdm import tqdm
from src.utilities import get_device


L_2_PI = torch.log(torch.tensor(2 * torch.pi))

def ll(x, mu, sd):
    return -0.5 * (((x - mu) ** 2) / (sd ** 2) + 2 * torch.log(sd) + L_2_PI)

def marginalize(
        config, 
        models_dir=None,
        rdir=None,
    ):
    device = get_device()
    vqvae_model = config["vqvae"]
    _, _, testl = data_loaders(data_cnf=config["input_data"])
    data = config["input_data"]
    pcnn, model_name = load_pcnn(models_dir, config)
    rdir = f'{rdir}/{data["dataset"]}/{data["pixel_representation"]}/vqvae_{model_name}'

    D = data["input_shape"]["height"]**2
    K = vqvae_model["codebook_size"]
    H = vqvae_model["latent_shape"]["height"]

    batch_size=2**16
    lls = []
    progress_bar = tqdm(total=(K**(H**2))//batch_size)
    
    for z_batch in tqdm(pcnn.cartesian_product_batch()):
        params = pcnn.vqvae.params(z_batch)
        progress_bar.update(1)
        params["mean"] = params["mean"].unsqueeze(0) 
        params["sd"] = params["sd"].unsqueeze(0) 
        lls.append(torch.cat([ll(bx.unsqueeze(1).to(device), params["mean"], params["sd"]) for bx, _ in testl]))
    
    lls = torch.stack(lls, dim=1)
    _bpd = -lls.logsumexp(dim=1).div(D * torch.log(torch.tensor(2)))
    f=bpd_formula(data)
    bpd = f(_bpd).mean().item()
    print(bpd)
    os.makedirs(rdir, exist_ok=True) 
    with open(f"{rdir}/mar.txt", 'w') as f:
        print(bpd, file=f)
    
    
if __name__ == '__main__':
    import os
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"    
    
    from recreate import arguments
    args = arguments()

    with open(args.config, 'r') as json_file:
        config = json.load(json_file)
    marginalize(
        config, 
        models_dir="/home/hadziarm/vqvae-tpm/experimental_settings/models",
        rdir="/home/hadziarm/vqvae-tpm/evaluations"
    )


    