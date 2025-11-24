import torch
import os
from tqdm import tqdm
import pandas as pd
import statistics
from torchvision.utils import save_image
from src.utilities import get_device, display_histogram, to_rgb, seed_everything
from src.mm.CategoricalMixture import CategoricalMixture as CMM
from src.mm.GaussianMixture import GaussianMixture as GMM
from datasets.data import data_loaders
from src.scripts.model_loaders import ModelLoader, load_dm, load_theta, load_pcnn
from src.metrics import bpd_dataset, fid_comparison, fid_score, fid_comparison_labeled
import torch.nn.functional as F

def images(dm, config, rdir):
    pr = config["input_data"]["pixel_representation"]
    pixel = pr["type"]
    n = 6**2
    if pixel == "cat":
        p = dm.batches[0]
        v = p.argmax(dim=1, keepdim=True)
        s = v / 255.
        weights = torch.arange(p.shape[1], device=p.device).view(1, -1, 1, 1)
        mean = torch.sum(p * weights, dim=1, keepdim=True) / 255.
        save_image(s[:n], f"{rdir}/mode-{dm.n_components}.png", nrow=6)
    elif pixel == "con":
        mean = torch.cat([m[0] for m, _ in dm.batches], dim=0)[:n].cpu()

    save_image(mean, f"{rdir}/mean-{dm.n_components}.png", nrow=6)

    torch.use_deterministic_algorithms(False)
    samples = dm.sample(n)
    torch.use_deterministic_algorithms(True, warn_only=True)
    save_image(samples, f"{rdir}/samples-{dm.n_components}.png", nrow=6)


def performance(models_dir, config, rdir, seeds, mode="test"):
    data = config["input_data"]
    trainl, validl, testl = data_loaders(data)
    datal = {
        "train": trainl,
        "valid": validl,
        "test": testl,
    }[mode]

    if config["input_data"]["supervised"]["use_labeling"]:
        fid_comp = fid_comparison_labeled
    else:
        fid_comp = fid_comparison
    
    device = get_device()
    records = []
    N = -1 # all data
    def safe_var(xs):
        return statistics.variance(xs) if len(xs) > 1 else 0.0
    print(seeds)
    for i in range(14, -1, -1):
        n = 2**i
        bpds, fids = [], []
        totp = None
        for seed in seeds:
            seed_everything(seed=seed)
            dm = GMM(load_theta(models_dir, config, seed=seed, n=n)[0])
            dm = dm.eval()
            totp = dm.total_parameters()
            if i == 14:
                images(dm, config, rdir)
                inpaints(dm, testl, device, rdir)
            bpds.append(bpd_dataset(dm, datal, data, device))

            fids.append(fid_comp(dm, datal, N, device))

        record = {
            "n_components": n,
            "n_par": totp,
            "BPD": statistics.mean(bpds),
            "BPD_var": safe_var(bpds),
            "FID": statistics.mean(fids),
            "FID_var": safe_var(fids),
        }
        records.append(record)
        
    df = pd.DataFrame(records)
    df = df.sort_values(by="n_components")
    csv_file = f"{rdir}/{mode}-performance-{len(seeds)}.csv"
    print(df)
    df.to_csv(csv_file, sep=',', index=False, encoding='utf-8')




def performance_full(models_dir, config, rdir, seeds, mode="test"):
    data = config["input_data"]
    trainl, validl, testl = data_loaders(data)
    datal = {
        "train": trainl,
        "valid": validl,
        "test": testl,
    }[mode]

    if config["input_data"]["supervised"]["use_labeling"]:
        fid_comp = fid_comparison_labeled
    else:
        fid_comp = fid_comparison
    
    device = get_device()
    records = []
    N = -1 # all data
    def safe_var(xs):
        return statistics.variance(xs) if len(xs) > 1 else 0.0
    print(seeds)
    for i in range(14, -1, -1):
        n = 2**i
        bpds, fids = [], []
        totp = None
        for seed in seeds:
            seed_everything(seed=seed)
            dm = GMM(load_theta(models_dir, config, seed=seed, n=n)[0])
            dm = dm.eval()
            totp = dm.total_parameters()
            if i == 14:
                images(dm, config, rdir)
                inpaints_full(dm, testl, device, rdir)
            bpds.append(bpd_dataset(dm, datal, data, device))

            fids.append(fid_comp(dm, datal, N, device))

        record = {
            "n_components": n,
            "n_par": totp,
            "BPD": statistics.mean(bpds),
            "BPD_var": safe_var(bpds),
            "FID": statistics.mean(fids),
            "FID_var": safe_var(fids),
        }
        records.append(record)
        
    df = pd.DataFrame(records)
    df = df.sort_values(by="n_components")
    csv_file = f"{rdir}/{mode}-performance-{len(seeds)}.csv"
    print(df)
    df.to_csv(csv_file, sep=',', index=False, encoding='utf-8')



def component_util(dm, loader, rdir, device):
    n = dm.n_components
    hist = torch.zeros(n).to(device)
    for bx in tqdm(loader):
        indices = dm.max_ll_component(bx.to(device)).to(device)
        hist.scatter_add_(0, indices, torch.ones_like(indices, dtype=hist.dtype).to(device))

    display_histogram(
        hist,
        title="Mixture component utilization",
        suptitle= f"Histogram of DM MNIST",
        xlabel= "Component index",
        ylabel= "Frequency",
        file_name=f"{rdir}/histogram_dm_{n}.png",
    )



def _square(d, e, B, device):
    # (B, C, H, W) with a central square set to 0
    tensor = torch.ones((B, e, d, d), device=device)
    start_idx = (d // 2) // 2
    end_idx = start_idx + d // 2
    tensor[:, :, start_idx:end_idx, start_idx:end_idx] = 0
    return tensor

def masks(B, dims, device):
    C, H, W = dims
    return [
        # Left half
        torch.cat([
            torch.ones((B, C, H, W // 2), device=device),
            torch.zeros((B, C, H, W - W // 2), device=device)
        ], dim=3).bool(),

        # Right half
        torch.cat([
            torch.zeros((B, C, H, W // 2), device=device),
            torch.ones((B, C, H, W - W // 2), device=device)
        ], dim=3).bool(),

        # Top half
        torch.cat([
            torch.ones((B, C, H // 2, W), device=device),
            torch.zeros((B, C, H - H // 2, W), device=device)
        ], dim=2).bool(),

        # # Bottom half
        # torch.cat([
        #     torch.zeros((B, C, H // 2, W), device=device),
        #     torch.ones((B, C, H - H // 2, W), device=device)
        # ], dim=2).bool(),

        # Center square
        _square(H, C, B, device).bool(),
    ]


def inpaints(dm, testl, device, rdir):
    batch=6
    images = []
    start = 0
    # bx = next(iter(testl)).to(device)
    o = next(iter(testl))

    if type(o) is list:
        bx, _ = o
    else:
        bx = o
    bx = bx.to(device)
    _, C, H, W = bx.shape
    dims = (C, H, W)
    for mask in masks(batch, dims, device):
        images_to_inpaint = bx[start:start+batch]
        image = dm.inpaint(images_to_inpaint, mask)
        if isinstance(dm, CMM):
            if image.shape[1] != 1:
                image = image.argmax(dim=1, keepdim=True)
            image = image.float()
            image /= 255.
        if image.shape[1] == 1:
            image = to_rgb(image, mask).cpu()
        images.append(image.cpu())
        start+=batch
    inpainted = torch.cat(images, dim=0)
    save_image(inpainted, f"{rdir}/inpaints-{dm.n_components}.png", nrow=6)


def inpaints_full(dm, testl, device, rdir):
    print("Inpainting full...")
    batch=12
    images = []
    start = 0
    # bx = next(iter(testl)).to(device)
    o = next(iter(testl))

    if type(o) is list:
        bx, _ = o
    else:
        bx = o
    bx = bx.to(device)
    _, C, H, W = bx.shape
    dims = (C, H, W)
    for mask in masks(batch, dims, device):
        images_to_inpaint = bx[start:start+batch]
        image = dm.inpaint(images_to_inpaint, mask)
        if isinstance(dm, CMM):
            if image.shape[1] != 1:
                image = image.argmax(dim=1, keepdim=True)
            image = image.float()
            image /= 255.
        if image.shape[1] == 1:
            print(image.shape)
            image = to_rgb(image, mask).cpu()
            print(image.shape)
        images.append(image.cpu())
        start+=batch
    inpainted = torch.cat(images, dim=0)
    save_image(inpainted, f"{rdir}/inpaints-{dm.n_components}.png", nrow=12)




def eval_dm(
        config, 
        models_dir,
        rdir=None,
        dm=None,
    ):
    device = get_device()
    data = config["input_data"]
    trainl, _, testl = data_loaders(config["input_data"])

    if dm is None:
        dm, model_name = ModelLoader(models_dir, config).load_dm(seed=config["seed"])
    else:
        model_name = ModelLoader(models_dir, config).path_builder.continuous_dm_name()

    pixel = data["pixel_representation"]["type"]
    rdir = f'{rdir}/{data["dataset"]}/{pixel}/dm_{model_name}'
    os.makedirs(rdir, exist_ok=True)
    print(rdir)

    images(dm, config, rdir)
    inpaints_full(dm, testl, device, rdir)

    bpd = bpd_dataset(dm, testl, data, device)
    print(f"Test BPD: {bpd:.3f}")
    
    n = 10**4
    print(n)
    if data["supervised"]["use_labeling"]:
        fid_comp = fid_comparison_labeled
    else:
        fid_comp = fid_comparison

    fid = fid_comp(dm, testl, n, device)
    print(f"FID: {fid:.3f}")

    seeds = config.get("seeds", [0])
    seeds = [config["seed"]]
    # performance(models_dir, config, rdir, seeds=seeds, mode="test")
    performance_full(models_dir, config, rdir, seeds=seeds, mode="test")

    # performance(models_dir, config, rdir, data, seeds=seeds, mode="train")
