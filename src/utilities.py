import torch
import os
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from src.EinsumNetwork.distributions.NormalArray import NormalArray
from src.EinsumNetwork.distributions.BinomialArray import BinomialArray
from src.EinsumNetwork.distributions.CategoricalArray import CategoricalArray
from torch.utils.data import DataLoader, Subset
from datasets.SamplerDataset import SamplerDataset
### Einets

def get_dist_family(config):
    family_selector = {
        "con": NormalArray,
        "cat": CategoricalArray,
        "bin": BinomialArray
    }
    pixel = config["input_data"]["pixel_representation"]["type"]
    return family_selector[pixel]

def rb_path(models_dir, config):
    data_config = config["input_data"]
    pr = data_config["pixel_representation"]
    pixel = pr["type"]
    K = config["K"]
    I = config["I"]
    hp = config["structure"]["hyperparams"]
    seed = config["seed"]
    PREFIXES = {
        "poon_domingos": "pd_",
        "poon_domingos_vertical": "pdv_",
        "poon_domingos_horizontal": "pdh_"
    }
    prefix = PREFIXES[config["structure"]["type"]]
    pd_pcs = prefix + '_'.join([str(i) for i in hp["pcs"]])
    file_path = f'{models_dir}/{data_config["dataset"]}/{pixel}/einet/{pd_pcs}/K_{K}/I_{I}/seed_{seed}'
    model_name = f"{pixel}_{pd_pcs}_{K}_{I}_seed_{seed}"
    
    return file_path, model_name

###

def get_device():
    return'cuda' if torch.cuda.is_available() else 'cpu'

def mixture_type(pixel_rep):
    from src.mm.CategoricalMixture import CategoricalMixture as CMM
    from src.mm.GaussianMixture import GaussianMixture as GMM
    mt = {
        "con": GMM,
        "cat": CMM,
        "bin": CMM
    }
    return mt[pixel_rep]

def latent_dim(lat_shape):
    h, w, c = lat_shape["height"], lat_shape["width"], lat_shape["channels"]
    return f"lat_{h}_{w}_{c}_old"


def seed_everything(seed: int):
    import random, os, torch, numpy
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    numpy.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def set_determinism(mode=True):
    """
    Enable or disable deterministic behavior in PyTorch.

    Args:
        mode (bool): If True, enables deterministic operations.
                     If False, allows non-deterministic but faster algorithms.
    """
    import os
    import torch

    if mode:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        print("[Determinism] Enabled: deterministic algorithms and cuDNN settings.")
    else:
        # Allow nondeterministic algorithms for potentially faster execution
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)
        print("[Determinism] Disabled: non-deterministic behavior allowed.")


def dataloader_random_sample(datal, n_samples, batch_size=None, **dataloader_kwargs):
    """
    Create a new DataLoader with randomly sampled data from the original DataLoader.
    
    Args:
        datal: Original DataLoader
        n_samples: Number of samples to randomly select
        batch_size: Batch size for the new DataLoader (defaults to original batch_size)
        **dataloader_kwargs: Additional arguments to pass to DataLoader constructor
    
    Returns:
        DataLoader with randomly sampled subset of data
    """
    total_samples = len(datal.dataset)
    
    if n_samples >= total_samples:
        print(f"Warning: Requested {n_samples} samples but only {total_samples} available")
        n_samples = total_samples
        indices = range(total_samples)
    else:
        indices = random.sample(range(total_samples), n_samples)
        
    subset = Subset(datal.dataset, indices)
        
    # Inherit other properties from original DataLoader if not overridden
    new_dataloader_kwargs = {
        'batch_size': datal.batch_size if batch_size is None else batch_size,
        'shuffle': datal.shuffle if hasattr(datal, 'shuffle') else False,
        'num_workers': datal.num_workers if hasattr(datal, 'num_workers') else 0,
        'pin_memory': datal.pin_memory if hasattr(datal, 'pin_memory') else False,
        'drop_last': datal.drop_last if hasattr(datal, 'drop_last') else False,
    }
    new_dataloader_kwargs.update(dataloader_kwargs)    
    return DataLoader(subset, **new_dataloader_kwargs)


def dataloader_from_sampler_lazy(sampler, total_samples, batch_size=2**8, **dataloader_kwargs):
    """
    Create a DataLoader that generates samples on-demand (memory efficient).
    
    Args:
        sampler: Object with a .sample(n) method that returns tensor samples
        total_samples: Total number of samples in the dataset
        batch_size: Batch size for the DataLoader
        **dataloader_kwargs: Additional arguments to pass to DataLoader constructor
    
    Returns:
        DataLoader that generates samples on-demand
    """
    dataset = SamplerDataset(sampler, total_samples)
    
    default_kwargs = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': 0,  # Be careful with multiprocessing and samplers
        'pin_memory': False,
        'drop_last': False,
    }
    
    default_kwargs.update(dataloader_kwargs)
    
    return DataLoader(dataset, **default_kwargs)

## inpainting

def to_rgb(img, mask):
    mask = mask.expand(-1, 3, -1, -1)
    red = torch.tensor([196,4,4], device=img.device) / 255.
    blue = torch.tensor([4,196,196], device=img.device) / 255.
    red_mask = mask * red.view(1, 3, 1, 1)  # Red mask=1 (196,4,4)
    blue_mask = ~mask * blue.view(1, 3, 1, 1)  # Blue mask=0 (4,196,196)
    return (red_mask + blue_mask) * img.expand(-1, 3, -1, -1)

##

def matplotlib_to_tikz(fig, file):
    import tikzplotlib
    fig.tight_layout()
    fig.savefig(f"{file}.png")
    tikzplotlib.save(f"{file}.tex")

def get_iterator(iterator, verbose, name=""):
    return iterator if not verbose else tqdm(iterator, desc=name)


def cartesian(A, B):
    grid_A, grid_B = torch.meshgrid(A, B)
    return torch.stack([grid_A.flatten(), grid_B.flatten()], dim=1)

def cartesians(A, n):
    return torch.stack([g.flatten() for g in torch.meshgrid(*([A] * n))], dim=1)

def display_histogram(
        hist,  
        title="",
        suptitle="", 
        xlabel="",
        ylabel="",
        file_name="histogram.png"
    ):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(suptitle, fontsize=16)
    ax.bar(range(hist.shape[0]), hist, color='skyblue', edgecolor='black')
    ax.axhline(y=1/hist.shape[0], color='red', linestyle='--', linewidth=1.5)
    ax.axhline(y=hist.mean(), color='green', linestyle='--', linewidth=1.5)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    plt.tight_layout()
    plt.savefig(file_name)
    plt.close()


def plot_images(images, titles, fff):
    """Helper function to plot a list of images."""
    n = len(images)
    fig, axs = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axs = [axs]
    for ax, img, title in zip(axs, images, titles):
        im = ax.imshow(img, cmap='Greys_r', interpolation='nearest')
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()
    plt.savefig(fff)


def format_large_numbers(num):
    if num >= 1_000_000_000:
        return f"{num / 1_000_000_000:.2f} B"
    elif num >= 1_000_000:
        return f"{num / 1_000_000:.2f} M"
    elif num >= 1_000:
        return f"{num / 1_000:.2f} K"
    return str(num)

def format_float_numbers(num):
    return f"{round(num, 3):.6f}"

def empty_folder(folder_path):
    for filename in os.listdir(folder_path): 
        file_path = os.path.join(folder_path, filename)  
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                os.rmdir(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")
    print("Deletion done")

def plot_gaussian(mu, sd, row, device):
    from torch.distributions import Normal
    x = torch.linspace(-3, 3, 1000).to(device)
    pdfs = Normal(mu, sd).log_prob(x.unsqueeze(-1).unsqueeze(-1)).exp().cpu()

    plt.figure(figsize=(10, 6))
    for i in range(pdfs.shape[1]):
        if row is not None and i != row:
            continue
        for j in range(pdfs.shape[2]):
            plt.plot(x.cpu(), pdfs[:, i, j], alpha=0.5)
