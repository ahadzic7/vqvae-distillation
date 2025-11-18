from src.vqvae.VQVAE_cat import VQVAE_cat
from src.vqvae.VQVAE_con import VQVAE_con
from src.utilities import *
from src.scripts.train.setup.setup import trainer_setup, vqvae_trainer
from datasets.data import data_loaders, cluster_data
import json

import pandas as pd
import numpy as np

from src.scripts.model_loaders import ModelLoader
from src.scripts.model_savers import save_vqvae

def time_evaluation(device, t, seeds, trainl, validl, log_folder, config, ld, version, max_time):
    results = []

    for seed in seeds:
        seed_everything(seed)

        # Initialize trainer, Timer, and custom epoch timer
        trainer, retrieved_timer, epoch_timer = vqvae_trainer(log_folder, config["epochs"], ld, f"{version}/seed_{seed}", max_time=max_time)

        # Fit model
        vqvae = t(config).to(device)
        trainer.fit(vqvae, trainl, validl)

        # Epoch-level times
        train_times = epoch_timer.epoch_times
        val_times = epoch_timer.val_times
        epochs_trained = len(train_times)

        # Total times from Lightning Timer
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

    # Convert to DataFrame
    df = pd.DataFrame(results)
    df.to_csv(f"{log_folder}/timing_results.csv", index=False)

    # Per-seed summary
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
    summary.to_csv(f"vqvae_{ds}_timing_summary.csv", index=False)


def train_vqvae(config, models_dir):
    device = get_device()
    data = config["input_data"]
    pixel = data["pixel_representation"]["type"]
    t = VQVAE_con if pixel == "con" else VQVAE_cat

    trainl, validl, _ = data_loaders(config["input_data"])
    
    clustering_config = config.get("clustering", None)
    if clustering_config is not None and clustering_config["use_clustering"]:
        models_dir, datals = cluster_data(config, [trainl, validl])
        trainl, validl = datals

    bx = next(iter(trainl))
    print(bx.shape)
    
    vqvae = t(config).to(device)
    vqvae = vqvae.train()
    print(vqvae.encoder)
    print(vqvae.latent_shape)
    print(vqvae.decoder.training)
    # exit()
    
    x = next(iter(trainl))[:32].to(device)
    print(vqvae.encode(x).shape)
    print(vqvae.forward(x).shape)

    # Clean and simple!
    log_folder, name, version = ModelLoader(models_dir, config).get_training_paths("vqvae")

    trainer, _, _ = vqvae_trainer(log_folder, config["epochs"], name=name, version=version)
    # trainer = trainer_setup(log_folder, config["epochs"], name=name, version=version)
    trainer.fit(vqvae, trainl, validl)
    save_vqvae(models_dir, config, vqvae)
    
    return vqvae
