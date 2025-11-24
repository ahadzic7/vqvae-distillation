import torch
import os
import numpy as np
from torchvision.utils import save_image
from src.utilities import dataloader_random_sample
from datasets.data import data_loaders
import torch.nn.functional as F
from src.scripts.model_loaders import load_vqvae
from src.metrics import fid_score, perplexity_grid, plot_codebook_perplexity, mse_loss, nll_loss

def images(vqvae, config, rdir, testl):
    x = next(iter(testl))[:config["n_recons"]].to(vqvae.device)
    recs = vqvae.reconstruction(x)
    save_image(recs, f"{rdir}/reconstruction.png", nrow=8)

def process_image(i, img, device):
    c, h, w = img.shape  # Get number of channels, height, width
    img = img.to(device)
    # Create red and black lines based on image dimensions
    red_line_bottom = torch.zeros(c, 1, w).to(device)
    red_line_bottom[0, :, :] = 1
    red_line_right = torch.zeros(c, h + 1, 1).to(device)
    red_line_right[0, :, :] = 1

    black_bottom = torch.zeros(c, 1, w).to(device)
    black_right = torch.zeros(c, h + 1, 1).to(device)

    if i >= 14 and (i - 14) % 20 == 0 and i != 94:
        x = torch.cat((img, red_line_bottom), dim=1)
        x = torch.cat((x, red_line_right), dim=2)
    elif (i >= 4 and (i - 4) % 20 == 0) or i == 94:
        x = torch.cat((img, black_bottom), dim=1)
        x = torch.cat((x, red_line_right), dim=2)
    elif any(10 + 20 * k <= i <= 19 + 20 * k for k in range(4)):
        # Add a blank column to the right
        x = torch.cat((img, torch.zeros(c, h, 1).to(device)), dim=2)
        red_line_bottom2 = torch.zeros(c, 1, x.shape[2]).to(device)
        red_line_bottom2[0, :, :] = 1
        x = torch.cat((x, red_line_bottom2), dim=1)
    else:
        x = torch.cat((img, black_bottom), dim=1)
        x = torch.cat((x, black_right), dim=2)
    return x

def reconstructions(vqvae, config, rdir):
    device = vqvae.device
    data = config["input_data"]
    data["data_loaders"] = False
    _, _, test = data_loaders(data)
    classes = range(10)
    l = []
    mid = len(classes) // 2
    for cs in zip(classes[:mid], classes[mid:]):
        rs = []
        for c in cs:
            datapoints = test.get_by_label(target_label=c, max_count=5)
            x_l = torch.stack(datapoints).to(device)
            recon = vqvae.reconstruction(x_l)
            B, C, H, W = recon.shape
            rs.append(recon.reshape(2, 5, C, H, W))
        l.append(torch.cat(rs, dim=1))
    rgb_images = torch.cat(l, dim=0)  # shape: [10, 10, C, H, W]
    B, C, H, W = rgb_images.shape[0]*rgb_images.shape[1], rgb_images.shape[2], rgb_images.shape[3], rgb_images.shape[4]
    rgb_images = rgb_images.reshape(B, C, H, W)

    l = [process_image(i, img, device) for i, img in enumerate(rgb_images.unbind(dim=0))]
    padded_img = l[0]
    _, H_p, W_p = padded_img.shape

    rgb_images = torch.stack(l).reshape(10, 10, C, H_p, W_p).permute(2, 0, 3, 1, 4).reshape(1, C, 10 * H_p, 10 * W_p)
    save_image(rgb_images, f"{rdir}/reconstruction.png", nrow=1)
    
def diff(vqvae, testl, data):
    device = vqvae.device
    delta = 0
    size = 0
    pixel = data["pixel_representation"]["type"]
    if pixel == "con":
        for bx, _ in testl:
            bx = bx.to(device)
            mu, _ = vqvae(bx).chunk(2, dim=1)
            delta += F.mse_loss(bx, mu)
            size += bx.shape[0]
    print(size)
    return delta / size


def fid_comparison(vqvae, testl, n):
    baselinel = dataloader_random_sample(testl, n)
    device = vqvae.device
    recs = torch.cat([vqvae(bx.to(device)).chunk(2, dim=1)[0] for bx in baselinel])
    # Split recs into batches manually to avoid TensorDataset tuple wrapping
    def create_batches(tensor, batch_size):
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    recsl = list(create_batches(recs, baselinel.batch_size))

    return fid_score(recsl, baselinel, device)


def eval_vqvae(
        config, 
        models_dir=None,
        rdir=None,
    ):
    data = config["input_data"]
    
    vqvae, model_name = load_vqvae(models_dir, config)
    pixel = data["pixel_representation"]["type"]
    rdir = f'{rdir}/{data["dataset"]}/{pixel}/dm_{model_name}'
    print(rdir)
    os.makedirs(rdir, exist_ok=True) 

    # print(vqvae.encoder)
    # exit()
    
    _, _, testl = data_loaders(config["input_data"])
    if data["supervised"]["use_labeling"]:
        reconstructions(vqvae, config, rdir)
    else:
        images(vqvae, config, rdir, testl)
    
    if not os.path.exists(f"{rdir}/perplexity.grid"):
        grid = perplexity_grid(vqvae, testl)
        torch.save(grid, f"{rdir}/perplexity.grid")
    else:
        grid = torch.load(f"{rdir}/perplexity.grid", weights_only=False)

    fig = plot_codebook_perplexity(vqvae, grid)
    fig.savefig(f"{rdir}/perplexity.png")

    n = 10**4
    print(n)
    fid = fid_comparison(vqvae, testl, n)
    print(f"FID: {fid:.2f}")


    mse = mse_loss(vqvae, testl)
    print(f"MSE: {mse:.2f}")
    nll = nll_loss(vqvae, testl)
    print(f"NLL: {nll:.2f}")

    M = data["input_shape"]["height"] * data["input_shape"]["width"] * data["input_shape"]["channels"]
    Dlog2 = M * np.log(2)
    bpd = nll / Dlog2 + 8
    print(f"BPD: {bpd:.2f}")
