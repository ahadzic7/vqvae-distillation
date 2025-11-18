from src.classifier.Classifier import Classifier
import torch
import json
from src.scripts.train.setup.setup import trainer_setup, classifier_trainer
from datasets.data import data_loaders

def train_classifier(config, models_dir):
    data = config["input_data"]
    model = config["classifier"]
    trainl, validl, _ = data_loaders(data)
    
    classifier = Classifier(model)

    pixel = data["pixel_representation"]["type"]


    log_folder=f'{models_dir}/{data["dataset"]}/{pixel}/classifier'
    # trainer = trainer_setup(log_folder, config["epochs"], model["architecture"])
    trainer = classifier_trainer(log_folder, config["epochs"], name=model["architecture"], version="0")

    trainer.fit(classifier, trainl, validl)

    save = f'{log_folder}/{model["architecture"]}/checkpoints'
    torch.save(classifier, f'{save}/model.ckpt')
    with open(f'{save}/config.json', 'w') as json_file:
        json.dump(config, json_file, indent=4)
