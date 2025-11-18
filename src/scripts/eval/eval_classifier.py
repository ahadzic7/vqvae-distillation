import torch
from src.utilities import get_device
from pytorch_lightning import Trainer
from datasets.data import data_loaders
from src.scripts.model_loaders import load_classifier

def eval_classifier(
        config, 
        models_dir=None,
        rdir=None
    ):
    data = config["input_data"]
    _, _, testl = data_loaders(data)
    
    pixel_rep = data["pixel_representation"]

    classifier, model_name = load_classifier(models_dir, config)
    classifier.eval()
    
    trainer = Trainer(accelerator='auto')
    test_results = trainer.test(classifier, dataloaders=testl)
    print(f"Test accuracy: {test_results[0]['test_acc']:.2f}")

    device = get_device()
    data_cnf = config["input_data"]
    _, _, testl = data_loaders(data_cnf)

    

    