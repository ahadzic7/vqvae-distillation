from src.pcnn.ConditionedPixelCNN import ConditionedPixelCNN
from src.pcnn.PixelCNN import PixelCNN
from src.utilities import get_device
import torch
from src.scripts.train.setup.setup import pixelcnn_trainer
from datasets.data import data_loaders, cluster_data
from src.scripts.model_loaders import ModelLoader
from src.scripts.model_savers import save_pcnn
from src.scripts.model_loaders import load_vqvae

def train_pcnn(config, models_dir=None, vqvae=None):
    if vqvae is None:
        vqvae, _ = load_vqvae(models_dir, config)
    device = get_device()

    data_cnf = config["input_data"]
    trainl, validl, _ = data_loaders(config["input_data"])

    clustering_config = config.get("clustering", None)
    if clustering_config is not None and clustering_config["use_clustering"]:
        models_dir, datals = cluster_data(config, [trainl, validl])
        trainl, validl = datals
        ccc = torch.cat([bx for bx in trainl])
        print(f"Clustered training data shape: {ccc.shape}")

    if data_cnf["supervised"]["use_labeling"]:
        print("AAA")
        classes = torch.arange(data_cnf["supervised"]["n_classes"])
        pixel_cnn = ConditionedPixelCNN(vqvae, classes, config["pcnn"]).to(device)
    else:
        pixel_cnn = PixelCNN(vqvae, config["pcnn"]).to(device)

    # Clean and simple!
    log_folder, name, version = ModelLoader(models_dir, config).get_training_paths("pcnn")

    trainer, _, _ = pixelcnn_trainer(log_folder, config["epochs"], name=name, version=version)
    trainer.fit(pixel_cnn, trainl, validl)
    save_pcnn(models_dir, config, pixel_cnn)

    return pixel_cnn

