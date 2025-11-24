import os
import torch
import statistics
from src.utilities import get_device, seed_everything
from src.scripts.model_loaders import load_cm, ModelLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from datasets.data import data_loaders
from src.metrics import bpd_dataset, fid_comparison

from src.mm.GaussianMixture import GaussianMixture as GMM
from src.scripts.eval.eval_dm import images, inpaints, inpaints_full

def cm_to_mm(cm, n=2**14):
    cm.sampler.n_bins = n

    mu, sd = [], []
    z, _ = cm.sampler(seed=None)
    with torch.no_grad():
        for bz in tqdm(torch.split(z.to(cm.device), 2**10, dim=0)):
            out = cm.decoder._get_params(bz)
            mu.append(out[0])
            sd.append(out[1])

    theta = { "mean": torch.cat(mu, dim=0), "sd": torch.cat(sd, dim=0) }
    return GMM(theta)


def performance(cm, rdir, config, seeds, mode="test"):
    trainl, validl, testl = data_loaders(config["input_data"])
    datal = {
        "train": trainl,
        "valid": validl,
        "test": testl,
    }[mode]

    def safe_var(xs):
        return statistics.variance(xs) if len(xs) > 1 else 0.0

    print(seeds)
    N = -1
    records = []
    for i in range(14, 13, -1):
        n = 2**i
        totp = None
        bpds, fids = [], []
        for seed in seeds:
            seed_everything(seed)
            mm = cm_to_mm(cm, n)
            if i == 14:
                images(mm, config, rdir)
                inpaints(mm, testl, cm.device, rdir)
                
            bpds.append(bpd_dataset(mm, datal, config["input_data"], cm.device))
            fids.append(fid_comparison(mm, datal, N, cm.device))
            totp = mm.total_parameters()
        record = {
            "n_components": n,
            "n_par": totp,
            "BPD": statistics.mean(bpds),
            "BPD_var": safe_var(bpds),
            "FID": statistics.mean(fids),
            "FID_var": safe_var(fids),
        }
        records.append(record)
        print()
    df = pd.DataFrame(records)
    csv_file = f"{rdir}/{mode}-performance-{len(seeds)}.csv"
    print(df)
    df.to_csv(csv_file, sep=',', index=False, encoding='utf-8')


def eval_cm(
        config, 
        models_dir=None,
        rdir=None,
        cm=None
    ):
    device = get_device()
    data_cnf = config["input_data"]
    _, _, testl = data_loaders(data_cnf)

    if cm is None:
        cm, model_name = load_cm(models_dir, config)
    else:
        model_name = ModelLoader(models_dir, config).path_builder.cm_name()

    cm.n_chunks = 2**10
    cm.sampler.n_bins = 2**14

    print(cm.decoder.min_std)
    print(cm.decoder.max_std)

    # samples = cm.sample(6**2)
    # save_image(samples, "samples.png", nrow=6)
    # bpd = bpd_dataset(cm, testl, data_cnf, device)
    # print(f"BPD: {bpd:.3f}")
    # exit()
    mm = cm_to_mm(cm)
    # bpd = bpd_dataset(mm, testl, data_cnf, device)
    # print(f"BPD: {bpd:.3f}")

    pixel = data_cnf["pixel_representation"]["type"]
    rdir = f'{rdir}/{data_cnf["dataset"]}/{pixel}/cm_{model_name}'
    print(rdir)
    os.makedirs(rdir, exist_ok=True)

    images(mm, config, rdir)
    inpaints_full(mm, testl, device, rdir)

    # bpd = bpd_dataset(mm, testl, data_cnf, device)
    # print(f"BPD: {bpd:.3f}")
    # n = 10**4
    # print(n)
    # fid = fid_comparison(mm, testl, n, device)
    # print(f"FID: {fid:.3f}")

    seeds = config.get("seeds", [0])
    # seeds = [0]
    seeds = [config["seed"]]
    performance(cm, rdir, config, seeds, mode="test")
    # performance(cm, rdir, data_cnf, seeds, mode="train")
    
    # samplesl = dataloader_from_sampler_lazy(mm, total_samples=10_000)

    # classifier, model_name = load_classifier(models_dir, config)
    # classifier.eval()

    # hist = classifier.classification_hist_dl(samplesl, True)
    # print(hist)




# def component_util(cm, loader, rdir, device):
#     n = cm.n_samples
#     hist = torch.zeros(n).to(device)
#     for bx in tqdm(loader):
#         indices = cm.max_log_prob_component(bx.to(device)).to(device)
#         hist.scatter_add_(0, indices, torch.ones_like(indices, dtype=hist.dtype).to(device))

#     fig, ax = plt.subplots(figsize=(10, 4))

#     fig.suptitle(f"Histogram of CM MNIST", fontsize=16)
#     file_name = f"{rdir}/histogram_cm_{n}.png"
   
#     ax.bar(range(n), hist.cpu(), color='skyblue', edgecolor='black')
#     ax.set_xlabel("Component index")
#     ax.set_ylabel("Frequency")
#     ax.set_title("Mixture component utilization")

#     plt.tight_layout()
#     plt.savefig(file_name)
#     plt.close()

# def model_size_vs_performance(cm, rdir, testl, data, seeds, device):
#     records = []
#     bx = next(iter(testl))
#     feature_dim = bx.numel() // bx.size(0)
#     totp = cm.total_parameters(feature_dim)
#     for i in range(15):
#         sum_bpd = 0.
#         cm.sampler.n_bins = 2**i
#         mm = cm_to_mm(cm)
#         for seed in seeds:
#             seed_everything(seed=seed)
#             avg_bpd += bpd_dataset(mm, testl, data, device)
        
#         record = {
#             "n_components": cm.sampler.n_bins,
#             "n_par": totp,
#             "BPD": sum_bpd / len(seeds),
#         }
#         records.append(record)

#     df = pd.DataFrame(records)
#     print(df)
#     csv_file = f"{rdir}/model_size_vs_performance.csv"
#     df.to_csv(csv_file, sep=',', index=False, encoding='utf-8')

# def plot_mpc_pdf(cm, component, rdir, device, row=None):
#     print(cm.decoder.mu_activation)
#     print(f"STD in [{cm.decoder.min_std:.3f}, {cm.decoder.max_std:.3f}]")
#     z, _ = cm.sampler(seed=42)
#     print(z.shape)
#     with torch.no_grad():
#         mu, sd = cm.decoder.forward_pass(z[component].unsqueeze(0).to(device))
#     print(f"lvar in [{sd.min().sqrt().log():.6f}, {sd.max().sqrt().log():.6f}]")
        
#     # plot_gaussian(mu[0,0], sd[0,0], row, device)
#     plt.title('Visualization of Gaussian PDFs')
#     plt.xlabel('x')
#     plt.ylabel('Probability Density')
#     plt.grid(alpha=0.3)
#     plt.savefig(f"{rdir}/MPC_gaussians.png", format="png")
