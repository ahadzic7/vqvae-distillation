import torch
from torchvision import transforms
from torchvision.datasets import MNIST, FashionMNIST, SVHN, CelebA
from torch.utils.data import DataLoader
import torch.nn.functional as F
from joblib import dump, load
import random
import math
import os
from tqdm import tqdm

from datasets.UnsupervisedDataset import UnsupervisedDataset
from datasets.BatchFileLoader import BatchFileLoader
from torch.utils.data import random_split


class OneHotEncode:
    def __init__(self, num_classes=256):
        self.num_classes = num_classes
    
    def __call__(self, tensor):       
        one_hot = F.one_hot(tensor.to(torch.int64), num_classes=self.num_classes)
        return one_hot.squeeze(dim=0).permute(2, 0, 1).float() 

    def inverse(self, one_hot, dim=0):
        if one_hot.shape[dim] != self.num_classes:
            raise ValueError(f"The {dim}. dimension of the input tensor must be equal to num_classes={self.num_classes}, but the size dimension was {one_hot.shape[dim]}.")
        return torch.argmax(one_hot, dim=dim).unsqueeze(dim)
     

class AddUniformNoise:
    """
        Converts a PIL image into a tensor, adds uniform noise in [0, 1) range.
        Reshapes the tensor to (CxHxW) (C = 1).
        Scales the pixel values to [0, 1) range.
    """
    def __call__(self, img):
        imgt= torch.tensor(list(img.getdata()), dtype=torch.uint8)
        imgt = imgt.view(*img.size, -1).permute(2, 0, 1).float()
        imgt += torch.empty_like(imgt).uniform_(0, 1)
        return imgt / 256


def pixel_value(trainl, data_cnf):
    pr = data_cnf["pixel_representation"]
    pixel = pr["type"]

    bx = next(iter(trainl))
    x = bx[0][0] if data_cnf["supervised"]["use_labeling"] else bx[0]
    msg = "Unknown pixel representation"
    print(x.shape)

    if x.shape[0] == 1:
        x_min, x_max = x.min(), x.max()
        if pixel == "con":
            msg = f"Pixel in [{x_min:.2f}, {x_max:.2f}]"
        elif pixel == "cat" or pixel == "bin":
            if pr["settings"][pixel]["onehot_encoded"]:
                msg = f"Pixel one hot encoded {{0, 1}}^{x.shape[0]}"
            else:
                msg = f"Pixel in {{{x_min}, ..., {x_max}}}"
    else:
        def format_rgb(fmt, x_min=None, x_max=None, shapes=None):
            """
            General RGB formatter.
            - fmt: a format string with placeholders.
            E.g. '{:.3f}–{:.3f}' or '{{{}, {}}}' or '{{0, 1}}^{}'
            - x_min, x_max: per-channel min/max (for value-based formats)
            - shapes: per-channel shapes (for one-hot formats)
            """
            if shapes:
                return (f"R: {fmt.format(shapes[0])}, G: {fmt.format(shapes[1])}, B: {fmt.format(shapes[2])}")
            return (f"R: {fmt.format(x_min[0], x_max[0])}, G: {fmt.format(x_min[1], x_max[1])}, B: {fmt.format(x_min[2], x_max[2])}")
        
        x_min, x_max = x.amin(axis=(1, 2)), x.amax(axis=(1, 2))
        if pixel == "con":
            msg = f"Pixel in [{format_rgb('{:.2f}–{:.2f}', x_min, x_max)}]"
        elif pixel == "cat" or pixel == "bin":
            if pr["settings"][pixel]["onehot_encoded"]:
                shapes = (x.shape[0], x.shape[1], x.shape[2])
                msg = f"Pixel one hot encoded [{format_rgb('{{0, 1}}^{{}}', shapes=shapes)}]"
            else:
                msg = f"Pixel in [{format_rgb('{{{}, ..., {}}}', x_min, x_max)}]"

    print(msg)


def mnist_like(data_cnf, t):
    d = "MNIST" if data_cnf["dataset"] == "BMNIST" else data_cnf["dataset"]
    ddir = f'./datasets/{d}'
    labeling = data_cnf["supervised"]["use_labeling"]

    datasets = {
        "MNIST": MNIST,
        "BMNIST": MNIST,
        "FMNIST": FashionMNIST,
    }
    ds = datasets[data_cnf["dataset"]]
    
    train_data = ds(root=ddir, train=True, download=True, transform=t)
    train = UnsupervisedDataset(train_data, labeling=labeling)
    train, valid = torch.utils.data.random_split(train, [50_000, 10_000])

    test_data = ds(root=ddir, train=False, download=True, transform=t)
    test = UnsupervisedDataset(test_data, labeling=labeling)

    return train, valid, test

def svhn(data_cnf, t, split_pct=.95):
    ddir = f'./datasets/{data_cnf["dataset"]}'
    labeling = data_cnf["supervised"]["use_labeling"]
    train_data = SVHN(root=ddir, split='train', download=True, transform=t)
    test_data = SVHN(root=ddir, split='test', download=True, transform=t)

    ts = int(split_pct * len(train_data))
    vs = len(train_data) - ts
    
    train, valid = torch.utils.data.random_split(train_data, [ts, vs])
    
    train = UnsupervisedDataset(train, labeling=labeling)
    valid = UnsupervisedDataset(valid, labeling=labeling)
    test = UnsupervisedDataset(test_data, labeling=labeling)
    return train, valid, test

def celeba(data_cnf, t):
    ddir = f'./datasets'
    labeling = data_cnf["supervised"]["use_labeling"]
    # train_data = CelebA(ddir, split='train', download=False, transform=t)
    # valid_data = CelebA(ddir, split='valid', download=False, transform=t)
    # test_data = CelebA(ddir, split='test', download=False, transform=t)
    data = CelebA(ddir, split='all', download=False, transform=t)
    # raw_images = data.data

    total = len(data)  # ~202,599
    valid_size = 10_000
    test_size = 10_000
    train_size = total - (valid_size + test_size)

    train_data, valid_data, test_data = random_split(
        data, [train_size, valid_size, test_size]
    )

    train = UnsupervisedDataset(train_data, labeling=labeling)
    valid = UnsupervisedDataset(valid_data, labeling=labeling)
    test = UnsupervisedDataset(test_data, labeling=labeling)
    return train, valid, test


def data_transformation_celeba(data_cnf):
    shp = data_cnf["input_shape"]
    trs = [
        transforms.CenterCrop(178),
        transforms.Resize((shp["height"], shp["width"])),
    ]
    pr = data_cnf["pixel_representation"]
    pixel = pr["type"]
    settings = pr["settings"][pixel]

    if pixel == "con":
        if settings["jittering"]:
            trs.append(AddUniformNoise())
        else:
            trs.append(transforms.ToTensor())
        
        if settings["normalize"]:
            C = data_cnf["input_shape"]["channels"]
            nc = [0.5 for _ in range(C)]
            print(f"Normalizing with {C}")
            trs.append(transforms.Normalize(nc, nc))
    else:
        raise Exception("Are we moving to celeba CAT?")

    trs = transforms.Compose(trs)
    return trs

def data_transformation(data_config):
    trs = []
    pr = data_config["pixel_representation"]
    pixel = pr["type"]
    settings = pr["settings"][pixel]

    if data_config["dataset"] == "celeba":
        return data_transformation_celeba(data_config)

    if pixel == "cat":
        trs.append(transforms.PILToTensor())
        if data_config["batch_size"]:
            trs.append(OneHotEncode(num_classes=256))

    elif pixel == "bin":
        trs.append(transforms.ToTensor())
        trs.append(transforms.Lambda(lambda x: (x > 0.5).float()))

    elif pixel == "con":
        if settings["jittering"]:
            trs.append(AddUniformNoise())
        else:
            trs.append(transforms.ToTensor())    

        if settings["normalize"]:
            C = data_config["input_shape"]["channels"]
            nc = [0.5 for _ in range(C)]
            print(f"Normalizing with {C}")
            trs.append(transforms.Normalize(nc, nc))
    else:
        raise Exception(f"Bad pixel repsresentation marker, recieved {pixel}")
    return transforms.Compose(trs)

def data_loaders(data_cnf):    
    dataset_factories = {
        "celeba": celeba,
        "SVHN": svhn,
        "MNIST": mnist_like,
        "FMNIST": mnist_like,
        "BMNIST": mnist_like,
    }
    factory = dataset_factories[data_cnf["dataset"]]
    t = data_transformation(data_cnf)
    train, valid, test = factory(data_cnf, t)

    batch_size = data_cnf["batch_size"]
    num_workers = data_cnf.get("num_workers", 2)

    print("A")

    trainl = DataLoader(train, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=False, pin_memory=True)
    validl = DataLoader(valid, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    testl = DataLoader(test, batch_size=batch_size, num_workers=num_workers, shuffle=False, pin_memory=True)
    
    pixel_value(trainl, data_cnf)

    print("B")
    
    return trainl, validl, testl



def cluster_filter(datal, config, n_clusters, cluster_id):
    models_dir_root = "models_clustering"
    ds = config['input_data']['dataset']
    cluster_dir = f"{models_dir_root}/{ds}/K_{n_clusters}"
    kmeans = load(f"{cluster_dir}/kmeans.joblib")
    d = []
    og_len = 0
    for bx in tqdm(datal):
        og_len += bx.shape[0]
        X_batch = bx.reshape(bx.size(0), -1).numpy()
        preds = kmeans.predict(X_batch)
        mask = (preds == cluster_id)
        if mask.sum() > 0:
            d.append(bx[mask])
    d = torch.cat(d, dim=0)
    print(f"Cluster {cluster_id}: {d.shape[0]} samples out of {og_len} ({100*d.shape[0]/og_len:.2f}%")

    if isinstance(datal, DataLoader):
        new_datal = DataLoader(d, batch_size=datal.batch_size, num_workers=datal.num_workers, shuffle=True, drop_last=datal.drop_last)
    elif isinstance(datal, BatchFileLoader):
        print("Warning: entire cluster loaded into memory, not using BatchFileLoader!")
        new_datal = DataLoader(d, batch_size=200, num_workers=2, shuffle=True, drop_last=False)
    return new_datal

def cluster_data(config, datals):
    models_dir_root = "models_clustering"
    ds = config['input_data']['dataset']
    clustering_config = config["clustering"]

    n_clusters = clustering_config["n_clusters"]
    cluster_id = clustering_config["cluster_id"]

    models_dir = f"{models_dir_root}/{ds}/K_{n_clusters}/cluster_{cluster_id}"

    for i, d in enumerate(datals):
        datals[i] = cluster_filter(d, config, n_clusters, cluster_id)
    print(len(datals))
    return models_dir, datals
