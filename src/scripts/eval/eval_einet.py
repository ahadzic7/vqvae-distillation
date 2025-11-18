import torch
import os
import statistics
from tqdm import tqdm
from datasets.data import data_loaders
from src.utilities import get_device, get_dist_family, seed_everything
from torchvision.utils import save_image
from src.scripts.model_loaders import load_einet
from src.EinsumNetwork.distributions.NormalArray import NormalArray
from src.metrics import eval_ll, fid_score
from src.utilities import dataloader_random_sample, dataloader_from_sampler_lazy
from src.metrics import bpd_dataset, fid_comparison, fid_score, fid_comparison_labeled
import pandas as pd
from src.scripts.eval.eval_dm import masks

def rb_path_tmp(config):
    data_config = config["input_data"]
    pr = data_config["pixel_representation"]
    pixel = pr["type"]
    print("Temporary rb function replace with original")
    if config["structure"]["type"] == "poon_domingos":
        hp = config["structure"]["hyperparams"]
        pd_pcs = "pd_" + '_'.join([str(i) for i in hp["pcs"]])
        return f'../models_v2/{pixel}/{data_config["dataset"]}/einet/{pd_pcs}/K_{config["K"]}/I_{config["I"]}'
    
    return "."


def inpainting(spn, data, dims, m):
    """
    Inpainting for EinsumNetwork SPN.

    Args:
        spn: trained SPN whose leaf has num_dims == 3 (RGB per pixel)
        data: (B, C, H*W)  # accepted input shape
              (B, C*H*W) is also accepted and reshaped internally
        dims: (H, W, C)
        m: mask: (H, W) or (C, H, W)
           True = observed, False = missing.
           With num_dims=3, marginalization is per-pixel, so if m is (C,H,W),
           a pixel is considered observed only if ALL its channels are observed.

    Returns:
        Tensor of shape (B, C, H, W)
    """
    H, W, C = dims
    B = data.shape[0]

    if data.dim() == 2:
        # (B, C*H*W) -> (B, C, H*W)
        assert data.shape[1] == C * H * W, f"Expected {C*H*W}, got {data.shape[1]}"
        data = data.view(B, C, H * W)
    else:
        assert data.shape == (B, C, H * W), f"Expected (B, {C}, {H*W}), got {tuple(data.shape)}"

    dev = next(spn.parameters()).device
    if data.device != dev:
        data = data.to(dev)
    if m.device != dev:
        m = m.to(dev)

    if m.dim() == 2 and m.shape == (H, W):
        m_pix = m.bool()  # (H, W)
    elif m.dim() == 3 and m.shape == (C, H, W):
        # observed iff ALL channels observed at that pixel
        m_pix = m.bool().all(dim=0)  # (H, W)
    else:
        raise ValueError(f"Mask must be (H,W) or (C,H,W), got {tuple(m.shape)}")

    marginalize_idx = []
    for i in range(H):
        base = i * W
        for j in range(W):
            if not m_pix[i, j]:
                marginalize_idx.append(base + j)

    X = H * W
    if marginalize_idx:
        mi, ma = min(marginalize_idx), max(marginalize_idx)
        assert 0 <= mi < X and 0 <= ma < X, f"marginalize_idx out of range: [{mi}, {ma}] / {X}"

    spn.set_marginalization_idx(marginalize_idx)

    # --- SPN expects (B, X, I) with I=3. Convert (B, C, H*W) -> (B, H*W, C)
    x_spn = data.permute(0, 2, 1).contiguous()  # (B, H*W, C)
    mpe = spn.mpe(x=x_spn)  # expect (B, H*W, C)

    if len(mpe.shape)==2:
        mpe = mpe.unsqueeze(1)
    mpe = mpe.permute(0, 2, 1).contiguous().view(B, C, H, W)
    return mpe

def data_sanity_check(data, dims, result_path):
    save_image(data.reshape(-1, dims[2], dims[0], dims[1]), f"{result_path}/data.png", nrow=6)
    print(data.shape)


def images(einet, rdir):
    n_samples=6**2
    samples = einet.sample(n_samples)
    file = f"{rdir}/samples.png"
    save_image(samples.cpu(), file, nrow=6)


def to_rgb(img, mask):
    # Ensure mask is float (0 or 1)
    mask = mask.float().permute(0, 2, 3, 1).repeat(1, 1, 1, 3)
    # Define colors
    red = torch.tensor([196, 4, 4], device=img.device) / 255.
    blue = torch.tensor([4, 196, 196], device=img.device) / 255.
    
    # Apply masks
    red_mask = mask * red.view(1, 1, 1, 3)
    blue_mask = (1 - mask) * blue.view(1, 1, 1, 3)
    
    return (red_mask + blue_mask) * img.permute(0, 2, 3, 1).expand(-1, -1, -1, 3)


def inpaints(einet, testl, dims, rdir, device="cuda", batch=12):
    inpaints = []
    for i, m in enumerate(masks(batch, dims=(dims[2], dims[0], dims[1]), device=device)):
        inpt = next(iter(testl))[batch*i:batch*(i+1)].reshape(batch, dims[2], -1).to(device)
        print(inpt.shape)
        inpt = inpainting(einet, inpt, dims, m = m[0].squeeze())
        print(inpt.shape)
        inpt_rgb = to_rgb(inpt, m) if dims[2] == 1 else inpt.permute(0, 2, 3, 1)
        print(inpt_rgb.shape)
        inpaints.append(inpt_rgb)
        print()
    
    inpaints = torch.cat(inpaints).cpu().permute(0, 3, 1, 2)
    print(inpaints.shape)
    file = f"{rdir}/inpaints.png"
    save_image(inpaints, file, nrow=12)


def performance(einet, config, rdir, dims, mode="test"):
    data = config["input_data"]
    trainl, validl, testl = data_loaders(data)
    datal = {
        "train": trainl,
        "valid": validl,
        "test": testl,
    }[mode]
    family = get_dist_family(config)

    if config["input_data"]["supervised"]["use_labeling"]:
        fid_comp = fid_comparison_labeled
    else:
        fid_comp = fid_comparison
    
    device = get_device()
    records = []
    N = -1 # all data

    def safe_var(xs):
        return statistics.variance(xs) if len(xs) > 1 else 0.0

    totp = einet.get_n_params()
    _, bpd = eval_ll(einet, datal, family, dims, device)
    fid = fid_comp(einet, datal, N, device)

    bpds, fids = [bpd.item()], [fid.item()]
    record = {
        "n_components": float('nan'),
        "n_par": totp,
        "BPD": statistics.mean(bpds),
        "BPD_var": safe_var(bpds),
        "FID": statistics.mean(fids),
        "FID_var": safe_var(fids),
    }
    records.append(record)
        
    df = pd.DataFrame(records)
    df = df.sort_values(by="n_components")
    csv_file = f"{rdir}/{mode}-performance-{0}.csv"
    print(df)
    df.to_csv(csv_file, sep=',', index=False, encoding='utf-8')



def eval_einet(
        config, 
        models_dir=None,
        rdir=None,
        einet=None,
    ):
    device = get_device()
    family = get_dist_family(config)
    _, _, testl = data_loaders(config["input_data"])
    
    data = config["input_data"]
    shp = data["input_shape"]
    dims = (shp["height"], shp["width"], shp["channels"])

    if einet is None:
        einet, model_name = load_einet(models_dir, config)
        if einet is None:
            exit("No model found!")
    else:
        _, model_name = load_einet(models_dir, config)

    einet.eval()
    einet = einet.to(device)

    pixel = data["pixel_representation"]["type"]
    rdir = f'{rdir}/{data["dataset"]}/{pixel}/einet_{model_name}'
    os.makedirs(rdir, exist_ok=True)

    images(einet, rdir)
    inpaints(einet, testl, dims, rdir)

    # ll, bpd = eval_ll(einet, testl, family, dims, device)
    # print(f"BPD: {bpd:.2f}")

    # N = 10**4
    # fid = fid_comparison(einet, testl, N, device)
    # print(f"FID: {fid:.3f}")

    performance(einet, config, rdir, dims)

# data_sanity_check(data, dims, rdir)