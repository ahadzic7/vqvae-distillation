import torch
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance
import matplotlib.pyplot as plt
from src.EinsumNetwork.distributions.NormalArray import NormalArray
from src.utilities import dataloader_random_sample, dataloader_from_sampler_lazy

## BPD

def bpd_formula(data_cnf):
    pr = data_cnf["pixel_representation"]
    pixel = pr["type"]
    if pixel == "bin" or pixel == "cat":
        return lambda bpd: bpd
    normalize = pr["settings"][pixel]["normalize"]
    jittering = pr["settings"][pixel]["jittering"]
    c = -1 if normalize else 0
    s = torch.tensor(256 if jittering else 255)
    const = torch.log(s) / torch.log(torch.tensor(2)) + c
    return lambda bpd: bpd + const

def bpd_dataset(mm, datal, data_cnf, device):
    f = bpd_formula(data_cnf)
    bpd = []
    labeling = data_cnf["supervised"]["use_labeling"]
    if labeling:
        for bx, _ in tqdm(datal):
            bpd.append(mm.bpd(bx.to(device), f))
    else:
        for bx in tqdm(datal):
            bpd.append(mm.bpd(bx.to(device), f))
    return torch.cat(bpd).mean().item()

def eval_ll(spn, datal, family, dims, device):
    def bits_per_dim(avg_ll, D, family):
        Dlog2 = D * np.log(2)
        bpp = (-avg_ll / Dlog2)
        if family == NormalArray:
            bpp += 8
        return bpp
    if family != NormalArray:
        target_shape = (-1, dims[0] * dims[1])
    else:
        target_shape = (-1, dims[0] * dims[1], dims[2])
    with torch.no_grad():
        ll, n = 0.0, 0
        print(target_shape)
        
        for bx in tqdm(datal):
            bx = bx.permute(0, 2, 3, 1).reshape(target_shape).to(torch.float)
            ll += spn.forward(bx.to(device)).squeeze().sum()
            n += bx.shape[0]
    avg_ll = ll/n
    return avg_ll, bits_per_dim(avg_ll, target_shape[1], family)

## FID

def get_inception_model(device, dims=2048):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx], normalize_input=False).to(device)
    model.eval()
    return model

def preprocess_images(images):
    _, C, H, W = images.shape
    if C == 1:
        images = images.repeat(1, 3, 1, 1)
    if H != 299 or W != 299:
        images = F.interpolate(images, size=(299, 299), mode='bilinear', align_corners=False)
    images = torch.clamp(images, 0.0, 1.0) * 2.0 - 1.0
    return images

@torch.no_grad()
def get_activations(model, datal, device):
    model.eval()
    features = []
    for bx in tqdm(datal, desc="Activations"):
    # for bx in datal:
        bx = preprocess_images(bx).to(device)
        pred = model(bx)[0]
        #if pred.size(2) != 1 or pred.size(3) != 1:
        pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        features.append(pred)
    return torch.cat(features, dim=0).squeeze().cpu().numpy()

@torch.no_grad()
def get_activations_labeled(model, datal, device):
    model.eval()
    features = []
    for batch in tqdm(datal, desc="Activations"):
        bx, by = batch
        bx = preprocess_images(bx).to(device)
        pred = model(bx)[0]
        #if pred.size(2) != 1 or pred.size(3) != 1:
        pred = F.adaptive_avg_pool2d(pred, output_size=(1, 1))
        features.append(pred)
    return torch.cat(features, dim=0).squeeze().cpu().numpy()

def fid_score(datal1, datal2, device):
    model = get_inception_model(device=device)
    act1 = get_activations(model, datal1, device)
    act2 = get_activations(model, datal2, device)
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

def fid_score_labeled(datal1, datal2, device):
    model = get_inception_model(device=device)
    act1 = get_activations_labeled(model, datal1, device)
    act2 = get_activations(model, datal2, device)
    mu1, sigma1 = np.mean(act1, axis=0), np.cov(act1, rowvar=False)
    mu2, sigma2 = np.mean(act2, axis=0), np.cov(act2, rowvar=False)
    return calculate_frechet_distance(mu1, sigma1, mu2, sigma2)

def fid_comparison(sampler, testl, n, device):
    if n == -1:
        baselinel = testl
    else:
        baselinel = dataloader_random_sample(testl, n)

    samplesl = dataloader_from_sampler_lazy(sampler, total_samples=len(baselinel.dataset))
    return fid_score(baselinel, samplesl, device)


def fid_comparison_labeled(sampler, testl, n, device):
    if n == -1:
        baselinel = testl
    else:
        baselinel = dataloader_random_sample(testl, n)

    samplesl = dataloader_from_sampler_lazy(sampler, total_samples=len(baselinel.dataset))
    return fid_score_labeled(baselinel, samplesl, device)


## codebook perplexity
def perplexity(histograms, eps=1e-12):
    norm = histograms.sum(dim=-1, keepdim=True) + eps
    ps = histograms / norm
    _ent = ps * ps.clamp_min(eps).log2()
    entropy = -_ent.sum(dim=-1)
    return 2 ** entropy

def perplexity_grid(vqvae_impl, datal):
    return perplexity(vqvae_impl.latent_cb_hists(datal))

def plot_codebook_perplexity(vqvae_impl, grid):
    H, W = grid.shape

    fig, ax = plt.subplots()
    im = ax.imshow(grid.cpu(), cmap='viridis', interpolation='nearest', aspect='auto')
    
    ax.set_xlim(-0.5, W-0.5)
    ax.set_ylim(H-0.5, -0.5)
    ax.set_xticks(np.arange(H))
    ax.set_yticks(np.arange(W))
    ax.set_xticklabels(np.arange(H))
    ax.set_yticklabels(np.arange(W))

    for i in range(H + 1):
        ax.axhline(y=i-0.5, color='black', linewidth=1.5)
        ax.axvline(x=i-0.5, color='black', linewidth=1.5)

    fig.colorbar(im, ax=ax)
    
    fig.suptitle(f'Latent perplexity - {H}x{W}')
    ax.set_title(f'max:{vqvae_impl.codebook_size} -- max used {grid.max().item():.2f}')
    
    return fig

## MSE 
def mse_loss(vqvae, datal):
    """ Compute MSE loss between original and reconstructed images in batches. """
    total_mse = 0.0
    for bx in tqdm(datal):
        bx = bx.to(vqvae.device)
        mode_rec = vqvae(bx).chunk(2, dim=1)[0]
        total_mse += F.mse_loss(mode_rec, bx, reduction='sum')
    return total_mse / len(datal.dataset)

def mse_loss_rec(vqvae_rec, datal):
    """ Compute MSE loss between original and reconstructed images in batches. """
    total_mse = 0.0
    for bx in tqdm(datal):
        bx = bx.to(vqvae_rec.device)
        total_mse += F.mse_loss(vqvae_rec(bx), bx, reduction='sum')
    return total_mse / len(datal.dataset)

## NLL 
def nll_loss(vqvae, datal):
    """Compute Gaussian NLL when decoder outputs mean and log-variance."""
    total_nll = 0.0
    for bx in tqdm(datal):
        bx = bx.to(vqvae.device)
        mu, logvar = vqvae(bx).chunk(2, dim=1)  # model must output mean and log-variance
        var = torch.exp(logvar)
        nll = 0.5 * (logvar + ((bx - mu) ** 2) / var + torch.log(torch.tensor(2 * torch.pi, device=bx.device)))
        total_nll += nll.sum()
    return total_nll / len(datal.dataset)

## classification kl div
def quality_comp(cls, basel, targetl, eps=1e-5, progress=True):
    p_bl = F.softmax(cls.classification_hist(basel, progress).clamp(min=eps).log(), dim=0)
    p_t  = F.softmax(cls.classification_hist(targetl, progress).clamp(min=eps).log(), dim=0)
    return F.kl_div(p_t.log(), p_bl, reduction='batchmean').item()
