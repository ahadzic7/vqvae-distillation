import os
import json
import csv
import matplotlib.pyplot as plt
import numpy as np
from pytorch_lightning.loggers import Logger
from pytorch_lightning.utilities import rank_zero_only
from typing import List, Dict, Optional, Tuple
import pytorch_lightning as pl

class UniversalLossPlotLogger(Logger):
    """
    A universal PyTorch Lightning logger that creates real-time loss plots
    for any model with configurable loss tracking.
    """
    
    def __init__(
        self,
        loss_groups: Dict[str, List[str]],
        save_dir: str,
        name: str = "training",
        version: Optional[str] = None,
        plot_frequency: int = 1,
        save_plots: bool = True,
        figsize: Tuple[int, int] = (12, 8),
        dpi: int = 150,
        save_best_model: bool = True,
        monitor_metric: str = "valid_loss",
        monitor_mode: str = "min"
    ):
        """
        Initialize the universal loss logger.
        
        Args:
            loss_groups: Dictionary mapping group names to lists of loss names.
                        Example: {
                            "Total Loss": ["train_loss", "valid_loss"],
                            "Reconstruction": ["train_rec", "valid_rec"],
                            "Regularization": ["train_kl", "valid_kl"]
                        }
            save_dir: Directory to save logs and plots
            name: Experiment name
            version: Version string/number
            plot_frequency: How often to update plots (every N epochs)
            save_plots: Whether to save plot files
            figsize: Figure size for plots
            dpi: DPI for saved plots
        """
        super().__init__()
        self._save_dir = save_dir
        self._name = name
        self._version = version or "0"
        self.plot_frequency = plot_frequency
        self.save_plots = save_plots
        self.figsize = figsize
        self.dpi = dpi

        self.save_best_model = save_best_model
        self.monitor_metric = monitor_metric
        self.monitor_mode = monitor_mode

        # Best model tracking
        self.best_metric_value = float('inf') if monitor_mode == 'min' else float('-inf')
        self.current_model = None
        
        # Validate and store loss groups
        self.loss_groups = loss_groups
        self.all_losses = set()
        for group_losses in loss_groups.values():
            self.all_losses.update(group_losses)
        
        # Create directories
        self.exp_log_dir = os.path.join(save_dir, name, self._version)
        self.checkpoint_dir = os.path.join(self.exp_log_dir, 'checkpoints')
        os.makedirs(self.exp_log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        # Storage for loss values: {loss_name: [(epoch, value), ...]}
        self.epoch_losses = {loss_name: [] for loss_name in self.all_losses}
        # Temporary storage within an epoch: {loss_name: [values]}
        self.step_losses = {loss_name: [] for loss_name in self.all_losses}
        
        self.current_epoch = 0
        self._last_step = 0
        
        self.data_file = os.path.join(self.exp_log_dir, "loss_data.csv")
        self.plot_file = os.path.join(self.exp_log_dir, "loss_plot.png")
        self.final_plot_file = os.path.join(self.exp_log_dir, "final_loss_plot.png")
        
        self.styles = {
            'train': ('-', "red"),
            'val': (':', "blue"),
            'valid': (':', "blue"),
            'test': ('--', "green")
        }
        
        self.latest_step_metrics = {}
        self.latest_epoch_metrics = {}


        print(f"Universal Loss Plot Logger initialized.")
        print(f"Tracking losses: {list(self.all_losses)}")
        print(f"Plots will be saved to: {self.exp_log_dir}")


        print(f"Universal Loss Plot Logger initialized.")
        print(f"Tracking losses: {list(self.all_losses)}")
        print(f"Monitoring '{monitor_metric}' ({'minimize' if monitor_mode == 'min' else 'maximize'}) for best model")
        print(f"Logs saved to: {self.log_dir}")
        print(f"Checkpoints saved to: {self.checkpoint_dir}")
        print("-------------------------------------------------------------")
        print("\n\n\n")

    @property
    def name(self) -> str:
        return self._name

    @property
    def version(self) -> str:
        return self._version

    @rank_zero_only
    def log_hyperparams(self, params: Dict):
        """Save hyperparameters to file."""
        hparams_file = os.path.join(self.exp_log_dir, "hparams.json")
        with open(hparams_file, 'w') as f:
            # Convert non-serializable objects to strings
            serializable_params = {}
            for k, v in params.items():
                try:
                    json.dumps(v)
                    serializable_params[k] = v
                except (TypeError, ValueError):
                    serializable_params[k] = str(v)
            json.dump(serializable_params, f, indent=2)

    @rank_zero_only
    def log_metrics(self, metrics: Dict, step: int):
        """Log metrics and collect them for plotting."""
        step_metrics = {k: v for k, v in metrics.items() if k.endswith("_step")}
        epoch_metrics = {k: v for k, v in metrics.items() if not k.endswith("_step")}

        if step_metrics:
            self.latest_step_metrics = step_metrics
        if epoch_metrics:
            self.latest_epoch_metrics = epoch_metrics

        for loss_name in self.all_losses:
            n = f"{loss_name}_step"
            if n in metrics:
                value = float(metrics[n])
                self.step_losses[loss_name].append(value)
        # Detect epoch end (step counter resets or decreases)
        if step < self._last_step:
            self._on_epoch_end()
        self._last_step = step

    def _on_epoch_end(self, latest_metrics: Optional[Dict] = None):
        """Process epoch-end aggregation and plotting."""
        current_epoch_metrics = {}
        for loss_name in self.all_losses:
            if self.step_losses[loss_name]:
                # Use mean aggregation
                epoch_value = np.mean(self.step_losses[loss_name])
                self.epoch_losses[loss_name].append((self.current_epoch, epoch_value))
                current_epoch_metrics[loss_name] = epoch_value
        self._save_data()
        
        self.step_losses = {loss_name: [] for loss_name in self.all_losses}
        self.current_epoch += 1

    def finalize(self, status: str):
        """Finalize logging and create final plots."""
        self._on_epoch_end()
        self._create_final_plots()
        print()
        print(f"Logging finalized with status: {status}")
        print()


    # csv table of losses
    def _save_data(self):
        """Save loss data to CSV file."""
        loss_names = list(self.epoch_losses.keys())
        with open(self.data_file, 'w', newline='') as f:
            writer = csv.writer(f)

            # Write header
            header = ["epoch"] + loss_names
            writer.writerow(header)

            # Find min length across all losses (safest)
            num_epochs = min(len(losses) for losses in self.epoch_losses.values())

            for epoch in range(num_epochs):
                row = [epoch]
                for loss_name in loss_names:
                    val = self.epoch_losses[loss_name][epoch]

                    # Unpack tuple (e.g. (0, np.float64(...)))
                    if isinstance(val, tuple) and len(val) > 1:
                        val = val[1]

                    # Convert numpy scalar → Python float
                    if isinstance(val, (np.generic,)):
                        val = float(val)
                        val = f"{val:.2f}"

                    row.append(val)
                writer.writerow(row)

    # plotting methods    
    
    def _format_loss_label(self, loss_name: str) -> str:
        """Format loss name into a readable label."""
        # Remove common prefixes and format
        formatted = loss_name
        for prefix in ['train_', 'val_', 'valid_', 'test_']:
            if formatted.startswith(prefix):
                formatted = formatted[len(prefix):]
                break
        
        # Capitalize and add prefix back as readable text
        if loss_name.startswith('train'):
            return f"Train {formatted.capitalize()}"
        elif loss_name.startswith(('val', 'valid')):
            return f"Val {formatted.capitalize()}"
        elif loss_name.startswith('test'):
            return f"Test {formatted.capitalize()}"
        else:
            return formatted.capitalize()

    def _get_loss_style(self, loss_name: str) -> Tuple[str, str]:
        """
        Determine color and line style for a loss based on its name.
        
        Returns:
            Tuple of (color, linestyle)
        """
        linestyle, color = self.styles["train"]
        for prefix, style in self.styles.items():
            if loss_name.startswith(prefix):
                linestyle, color = style
                break      
        return color, linestyle

    def _plot_losses(self):
        """
        Generalized function to plot training losses.

        Args:
            mode (str): 'update' for real-time plots, 'final' for high-quality final plots.
        """
        if not any(self.epoch_losses.values()):
            return
        n_groups = len(self.loss_groups)
        if n_groups == 0:
            return
        
        figsize = (14, max(6, 2 * n_groups))
        linewidth, markersize = 2.5, 5
        fig, axes = plt.subplots(n_groups, 1, figsize=figsize, sharex=True)
        if n_groups == 1:
            axes = [axes]
        
        for idx, (group_name, group_losses) in enumerate(self.loss_groups.items()):
            for loss_name in group_losses:
                if loss_name in self.epoch_losses and self.epoch_losses[loss_name]:
                    epochs, values = zip(*self.epoch_losses[loss_name])
                    color, linestyle = self._get_loss_style(loss_name)
                    label = self._format_loss_label(loss_name)
                    axes[idx].plot(
                        epochs, 
                        values,
                        color=color, 
                        linestyle=linestyle,
                        linewidth=linewidth, 
                        marker='o', 
                        markersize=markersize,
                        label=label
                    )
            
            axes[idx].set_ylabel("Loss", fontsize=14, fontweight="bold")
            axes[idx].set_title(group_name, fontsize=16, fontweight="bold")
            axes[idx].legend(fontsize=12)
            axes[idx].grid(True, alpha=0.3)
            axes[idx].spines['top'].set_visible(False)
            axes[idx].spines['right'].set_visible(False)

        axes[-1].set_xlabel("Epoch", fontsize=14, fontweight="bold")
        plt.tight_layout()
        return fig

    def _update_plots(self):
        if not self.save_plots:
            return
        fig = self._plot_losses()
        if fig is None:
            return 
        fig.savefig(self.plot_file, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        print()
        print(f"Plot updated at epoch {self.current_epoch}")
        print()
        plt.close(fig)

    def _create_final_plots(self):
        fig = self._plot_losses()
        if fig is None:
            return 
        fig.savefig(self.final_plot_file, dpi=self.dpi, bbox_inches="tight", facecolor="white")
        print()
        print()
        print(f"Final plots saved:\n  PNG: {self.final_plot_file}")
        plt.close(fig)


