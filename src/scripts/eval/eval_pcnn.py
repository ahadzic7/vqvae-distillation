import torch
from torchvision.utils import save_image
from src.utilities import *
from src.mm.CategoricalMixture import CategoricalMixture as CMM
from src.mm.GaussianMixture import GaussianMixture as GMM
import pandas as pd
import matplotlib.pyplot as plt
from datasets.data import data_loaders
from src.searches.minll_search import minll_search
from src.scripts.model_loaders import load_pcnn
from src.metrics import bpd_dataset, fid_score

def images(theta, config, rdir):
    pr = config["input_data"]["pixel_representation"]
    
    if pr == "cat":
        p = theta["probs"]
        n = p.shape[0]
        v = p.argmax(dim=1, keepdim=True)
        s = v / 255.
        weights = torch.arange(p.shape[1], device=p.device).view(1, -1, 1, 1)
        mean = torch.sum(p * weights, dim=1, keepdim=True) / 255.
        save_image(s, f"{rdir}/mode-{n}.png", nrow=10)
        MM = CMM
    elif pr == "con":
        mean = theta["mean"]
        n = mean.shape[0]
        mean = mean[:100]
        MM = GMM

    save_image(mean, f"{rdir}/mean-{n}.png", nrow=10)

    mm = MM(theta=theta)
    # samples = mm.sample(components=torch.arange(labels.shape[0]))
    samples = mm.sample(6**2)
    print(samples.shape)
    save_image(samples, f"{rdir}/samples-{n}.png", nrow=6)


def model_size_vs_performance(MM, pcnn, rdir, testl, data, seeds):
    device = pcnn.device
    tp = pcnn.count_parameters() + pcnn.vqvae.count_parameters()
    records = []
    e = 14
    theta = pcnn.params(2**e) 
    for i in range(e+1):
        n_components = 2**i
        sum_bpd = 0
        for seed in seeds:
            seed_everything(seed)
            t = {
                "mean": theta["mean"][:n_components],
                "sd": theta["sd"][:n_components]
            }
            mm = MM(theta=t)
            sum_bpd += bpd_dataset(mm, testl, data, device)

        record = {
            "# components": n_components,
            "# tr par": tp,
            "# MM par": mm.total_parameters(),
            "BPD": sum_bpd / len(seeds),
        }
        records.append(record)
        df = pd.DataFrame(records)
        csv_file = f"{rdir}/model_size_vs_performance.csv"
        print(csv_file)
        print(df)
        df.to_csv(csv_file, sep=',', index=False, encoding='utf-8')

  
def component_util(mm, loader, rdir, device):
    n = mm.n_components
    hist = torch.zeros(n).to(device)
    for bx in tqdm(loader):
        indices = mm.max_ll_component(bx.to(device)).to(device)
        hist.scatter_add_(0, indices, torch.ones_like(indices, dtype=hist.dtype).to(device))

    display_histogram(
        hist,
        title="Mixture component utilization",
        suptitle= f"Histogram of MM MNIST",
        xlabel= "Component index",
        ylabel= "Frequency",
        file_name=f"{rdir}/histogram_mm_{n}.png",
    )

def eval_pcnn(
        config, 
        models_dir=None,
        rdir=None,
    ):
    device = get_device()
    data = config["input_data"]
    trainl, _, testl = data_loaders(data)
    pcnn, model_name = load_pcnn(models_dir, config)
    MM = mixture_type(pixel_rep=config["input_data"]["pixel_representation"])
    
    rdir = f'{rdir}/{data["dataset"]}/{data["pixel_representation"]}/vqvae_{model_name}'
    os.makedirs(rdir, exist_ok=True) 
    ohe = data["onehot_encoded"]

    n_components = 2**14

    # theta = minll_search(pcnn, trainl, n_components)
    theta = pcnn.params(n_components) 
 
    for c in range(data["input_shape"]["channels"]):
        v = theta["sd"][:,c]
        print(f'Channel {c} sd in [{v.min():.2f}, {v.max():.2f}]')
    
    mm = MM(theta=theta)
    bpd = bpd_dataset(mm, testl, data, device)
    print(bpd)



