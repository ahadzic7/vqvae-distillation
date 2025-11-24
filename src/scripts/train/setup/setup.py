import pytorch_lightning as pl
from src.scripts.train.setup.UniversalPlotLogger import UniversalLossPlotLogger
from src.scripts.train.setup.UniversalLoggerCallback import UniversalLoggerCallback

def trainer_setup(log_folder, max_epochs, name="", version="", max_time="01:00:00:00"):
    cp_best_model_valid = pl.callbacks.ModelCheckpoint(
        save_top_k=1,
        monitor='valid_loss',
        mode='min',
        filename='best_{epoch}'
    )
    early_stop_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor="valid_loss",
        min_delta=0.00,
        patience=15,
        verbose=False,
        mode='min'
    )
    callbacks = [cp_best_model_valid, early_stop_callback]
    logger = pl.loggers.TensorBoardLogger(log_folder, name=name, version=version)
    
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        max_time=max_time,  # Add this parameter
        callbacks=callbacks,
        logger=logger,
        deterministic=True
    )
    return trainer

import time
class EpochTimer(pl.Callback):
    def __init__(self):
        super().__init__()
        self.epoch_times = []
        self.val_times = []
        self._epoch_start = None
        self._val_start = None

    def on_train_epoch_start(self, trainer, pl_module):
        self._epoch_start = time.time()

    def on_train_epoch_end(self, trainer, pl_module):
        elapsed = time.time() - self._epoch_start
        self.epoch_times.append(elapsed)

    def on_validation_start(self, trainer, pl_module):
        self._val_start = time.time()

    def on_validation_end(self, trainer, pl_module):
        elapsed = time.time() - self._val_start
        self.val_times.append(elapsed)


def trainer_with_universal_logger(
    log_folder: str,
    max_epochs: int,
    loss_groups = None,
    model_type= None,
    name: str = "experiment",
    version = None,
    max_time: str = "01:00:00:00",
    plot_frequency: int = 1,
    patience: int = 15,
    min_delta: float = 1e-3,
    monitor_metric: str = "valid_loss",
    **logger_kwargs
) -> pl.Trainer:
    """
    Create a PyTorch Lightning trainer with universal loss plotting.
    
    Args:
        log_folder: Directory to save logs and plots
        max_epochs: Maximum number of training epochs
        loss_groups: Dictionary mapping group names to loss names for plotting.
                    If None, model_type must be specified.
        model_type: Predefined model type ('vqvae', 'pixelcnn', 'vae').
                   Ignored if loss_groups is provided.
        name: Experiment name
        version: Version string/number
        max_time: Maximum training time
        plot_frequency: How often to update plots (every N epochs)
        patience: Early stopping patience
        min_delta: Minimum change for early stopping
        monitor_metric: Metric to monitor for checkpointing and early stopping
        **logger_kwargs: Additional arguments passed to UniversalLossPlotLogger
        
    Returns:
        Configured PyTorch Lightning trainer
    """
    assert loss_groups is not None, "'loss_groups' must be specified"
    
    logger = UniversalLossPlotLogger(
        loss_groups=loss_groups,
        save_dir=log_folder,
        name=name,
        version=version,
        plot_frequency=plot_frequency,
        **logger_kwargs
    )
    callback = UniversalLoggerCallback(
        logger, 
        logger.exp_log_dir, 
        patience, min_delta, 
        logger.monitor_mode
    )

    timer_callback = pl.callbacks.Timer()
    epoch_timer_callback = EpochTimer()

    trainer = pl.Trainer(
        max_epochs=max_epochs,
        max_time=max_time,
        callbacks=[callback],
        logger=logger,
        deterministic=True,
        log_every_n_steps=50,  # Reasonable default for epoch-based plotting
        check_val_every_n_epoch=1,
        enable_progress_bar=True,
        enable_checkpointing=False,
    )
    
    return trainer, timer_callback, epoch_timer_callback

def vqvae_trainer(
    log_folder: str,
    max_epochs: int,
    name: str = "vqvae_experiment",
    version:str = "",
    **kwargs
) -> pl.Trainer:
    """Create a trainer specifically configured for VQ-VAE models."""
    loss_groups = {
        "Total Loss": ["train_loss", "valid_loss"],
        "Reconstruction Loss": ["train_rec", "valid_rec"],
        "Codebook Loss": ["train_cod", "valid_cod"],
        "Commitment Loss": ["train_com", "valid_com"],
        # "Beta": ["beta"],
    }
    return trainer_with_universal_logger(
        log_folder=log_folder,
        max_epochs=max_epochs,
        loss_groups=loss_groups,
        model_type="vqvae",
        name=name,
        version=version,
        **kwargs
    )

def pixelcnn_trainer(
    log_folder: str,
    max_epochs: int,
    name: str = "pixelcnn_experiment",
    version:str = "",
    **kwargs
) -> pl.Trainer:
    """Create a trainer specifically configured for PixelCNN models."""
    loss_groups = { "Cross entropy Loss": ["train_loss", "valid_loss"], }
    return trainer_with_universal_logger(
        log_folder=log_folder,
        loss_groups=loss_groups,
        max_epochs=max_epochs,
        model_type="pixelcnn",
        name=name,
        version=version,
        **kwargs
    )

def cm_trainer(
    log_folder: str,
    max_epochs: int,
    name: str = "cm_experiment",
    version:str = "",
    **kwargs
) -> pl.Trainer:
    """Create a trainer specifically configured for ContinuousMixture models."""
    loss_groups = { "NLL": ["train_loss", "valid_loss"], }
    return trainer_with_universal_logger(
        log_folder=log_folder,
        loss_groups=loss_groups,
        max_epochs=max_epochs,
        model_type="cm",
        name=name,
        version=version,
        **kwargs
    )


# Usage examples and documentation
"""
USAGE EXAMPLES:

1. VQ-VAE Training (using predefined configuration):
   ```python
   trainer = vqvae_trainer(
       log_folder="./logs",
       max_epochs=100,
       name="my_vqvae_experiment"
   )
   
   model = VQVAE(...)  # Your VQ-VAE model
   trainer.fit(model, train_dataloader, val_dataloader)
   ```

2. Custom Model Training:
   ```python
   # Define your loss groups
   loss_groups = {
       "Total Loss": ["train_loss", "valid_loss"],
       "Adversarial Loss": ["train_adv", "val_adv"],
       "Perceptual Loss": ["train_perc", "val_perc"]
   }
   
   trainer = trainer_with_universal_logger(
       log_folder="./logs",
       max_epochs=200,
       loss_groups=loss_groups,
       name="gan_experiment",
       plot_frequency=5  # Update plots every 5 epochs
   )
   ```

3. PixelCNN Training:
   ```python
   trainer = pixelcnn_trainer(
       log_folder="./logs",
       max_epochs=150,
       name="pixelcnn_experiment",
       patience=20  # Custom early stopping patience
   )
   ```

The logger will automatically:
- Create real-time plots updated during training
- Save high-quality final plots (PNG and PDF)
- Store raw loss data in JSON format
- Handle proper epoch aggregation of step-wise losses
- Use appropriate colors and line styles for train/val losses
"""