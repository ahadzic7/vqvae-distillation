from src.cm.ContinuousMixture import ContinuousMixture
from src.cm.decoders import CategoricalDecoder, GaussianDecoder
from src.cm.utils.bins_samplers import GaussianQMCSampler
from src.cm.nets import mnist_conv_decoder, celeba_conv_decoder_128, svhn_conv_decoder
from src.utilities import latent_dim, seed_everything
from src.scripts.train.setup.setup import trainer_setup, cm_trainer
from datasets.data import data_loaders
from src.scripts.eval.eval_cm import performance

import torch.nn as nn

from src.utilities import get_device
import numpy as np
from src.scripts.model_savers import save_cm
from src.scripts.model_loaders import ModelLoader, load_cm
import os

def CM_args(d, s, model):
    return {
        'decoder': d,
        'sampler': s,
        'k': model["K"],
        'n_samples': model["n_bins"],
        'lat_dim': model["latent_shape"]["channels"]
    }

selector = {
    "MNIST": mnist_conv_decoder,
    "celeba": celeba_conv_decoder_128,
    "SVHN": svhn_conv_decoder,
}

import pandas as pd
import numpy as np

def process(results, seed, epoch_timer, retrieved_timer):
    train_times = epoch_timer.epoch_times
    val_times = epoch_timer.val_times
    epochs_trained = len(train_times)

    total_train_time = retrieved_timer.time_elapsed("train")
    total_val_time = retrieved_timer.time_elapsed("validate")

    for epoch in range(epochs_trained):
        results.append({
            "seed": seed,
            "epoch": epoch + 1,
            "train_time": train_times[epoch],
            "val_time": val_times[epoch] if epoch < len(val_times) else None,
            "cumulative_train_time": sum(train_times[:epoch+1]),
            "cumulative_val_time": sum(val_times[:epoch+1]) if epoch < len(val_times) else None
        })
    return results


def build_model(config, device):
    data = config["input_data"]
    model = config["model"]
    dnet = model["decoder_net"]

    pr = data["pixel_representation"]
    pixel = pr["type"]

    decoder_fun = selector[data["dataset"]]
    
    net = decoder_fun(
        latent_dim=model["latent_shape"]["channels"],
        n_filters=dnet["n_filters"],
        batch_norm=dnet["batch_norm"],
        learn_std=model["learn_std"],
        bias=dnet["bias"],
        resblock=dnet["resblock"],
        out_channels=dnet["out_channels"],
        n_layers=model["layers"]
    )

    if pixel == "cat":
        dec = CategoricalDecoder(net).to(device)
    elif pixel == "con":
        normalize = pr["settings"][pixel]["normalize"]
        act = nn.Tanh() if normalize else nn.Sigmoid()
        dec = GaussianDecoder(
            net, 
            learn_std=model["learn_std"],
            min_std=model["min_std"],
            max_std=model["max_std"],
            mu_activation = act
        ).to(device)
    C = model["latent_shape"]["channels"]
    sampler = GaussianQMCSampler(C, model["n_bins"])
    cm = ContinuousMixture(**CM_args(dec, sampler, model)).to(device)
    cm.missing = model["missing"]
    cm.n_chunks = model["n_chunks"]

    return cm


def build_trainer(config, models_dir):
    data = config["input_data"]
    model = config["model"]

    pr = data["pixel_representation"]
    pixel = pr["type"]

    ld = latent_dim(model["latent_shape"])
    version=f"n_bins-{model['n_bins']}"
    log_folder=f'{models_dir}/{data["dataset"]}/{pixel}/cm/layers_{model["layers"]}'
    max_time = config["max_time"] 
    save = f'{log_folder}/{ld}/{version}/checkpoints'

    # trainer = trainer_setup(log_folder, config["epochs"], ld, version, max_time=max_time)
    trainer, retrieved_timer, epoch_timer = cm_trainer(log_folder, config["epochs"], ld, version, max_time=max_time)
    return trainer, retrieved_timer, epoch_timer, save

def time_evaluation(seeds, config, models_dir, device):
    print(seeds)
    results = []
    for seed in seeds:
        seed_everything(seed=seed)
        trainl, validl, _ = data_loaders(config["input_data"])
        cm = build_model(config, device)
        trainer, retrieved_timer, epoch_timer, _ = build_trainer(config, models_dir)
        trainer.fit(cm, trainl, validl)
        results = process(results, seed, epoch_timer, retrieved_timer)

        df = pd.DataFrame(results)

        summary = df.groupby("seed").agg(
            total_train_time=pd.NamedAgg(column="train_time", aggfunc="sum"),
            total_val_time=pd.NamedAgg(column="val_time", aggfunc="sum"),
            epochs_trained=pd.NamedAgg(column="epoch", aggfunc="max"),
            avg_train_per_epoch=pd.NamedAgg(column="train_time", aggfunc="mean"),
            avg_val_per_epoch=pd.NamedAgg(column="val_time", aggfunc="mean"),
            train_val_ratio=pd.NamedAgg(column="train_time", aggfunc=lambda x: x.sum() / df.loc[x.index, "val_time"].sum())
        ).reset_index()

        ds = config["input_data"]["dataset"]
        print(summary)
        summary.to_csv(f"./timing_results/cm_{ds}_timing_summary_seed-{seed}.csv", index=False)



def train_cm(config, models_dir, rdir=None):
    trainl, validl, _ = data_loaders(config["input_data"])
    device = get_device()
    
    cm = build_model(config, device)

    # Clean and simple!
    loader = ModelLoader(models_dir, config)
    log_folder, name, version = loader.get_training_paths("cm")

    trainer, _, _ = cm_trainer(log_folder, config["epochs"], name=name, version=version)
    trainer.fit(cm, trainl, validl)
    save_cm(models_dir, config, cm)

    data_cnf = config["input_data"]

    pixel = data_cnf["pixel_representation"]["type"]
    cm, model_name = load_cm(models_dir, config)
    rdir = f'{rdir}/{data_cnf["dataset"]}/{pixel}/cm_{model_name}'
    print(rdir)
    os.makedirs(rdir, exist_ok=True)

    seeds = [config["seed"]]
    cm = cm.to("cuda")
    performance(cm, rdir, config, seeds)

    # seeds = [0]
    # time_evaluation(seeds, config, models_dir, device)
    # exit()
