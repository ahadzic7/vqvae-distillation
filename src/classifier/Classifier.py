import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from tqdm import tqdm
from src.classifier.Classifier_arch import arch_selector

class Classifier(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.num_classes = config["num_classes"]
        self.learning_rate = config["learning_rate"]
        
        self.network = arch_selector(config["architecture"])()
        self.save_hyperparameters(ignore=["network"])

    def forward(self, x):
        return self.network(x)
    
    def loss(self, batch):
        x, y = batch
        logits = self(x)
        return F.cross_entropy(logits, y), logits

    def training_step(self, batch, batch_idx):
        loss, logits = self.loss(batch)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch[1]).float().mean()
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('train_acc', acc, prog_bar=True, logger=True, on_step=True, on_epoch=True)

        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        val_loss, logits = self.loss(batch)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch[1]).float().mean()
        self.log('valid_loss', val_loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        self.log('valid_acc', acc, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return val_loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        test_loss, logits = self.loss(batch)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == batch[1]).float().mean()
        self.log("test_loss", test_loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)
        return test_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.network.parameters(), lr=self.learning_rate)
    
    @torch.no_grad()
    def classification_hist(self, samples, progress=False):
        l = tqdm(samples) if progress else samples
        return torch.cat([self(x.to(self.device)).argmax(dim=1) for x in l]).bincount(minlength=self.num_classes)

    @torch.no_grad()
    def classification_hist_dl(self, dataloader, progress=False):
        self.eval()  # set model to eval mode
        device = self.device
        hist = torch.zeros(self.num_classes, dtype=torch.long, device=device)
        l = tqdm(dataloader, desc="Computing class histogram") if progress else dataloader

        for x in l:
            preds = self(x.to(device)).argmax(dim=1)
            hist += torch.bincount(preds, minlength=self.num_classes)

        return hist.cpu()