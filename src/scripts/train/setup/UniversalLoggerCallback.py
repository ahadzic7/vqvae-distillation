import os, torch
from src.scripts.train.setup.UniversalPlotLogger import UniversalLossPlotLogger
import pytorch_lightning as pl

class UniversalLoggerCallback(pl.Callback):
    def __init__(
        self,
        logger: UniversalLossPlotLogger,
        save_dir: str,
        patience: int = 15,
        min_delta: float = 1e-5,
        mode: str = "min",
        check_finite: bool = True,
        save_weights_only: bool = False,
    ):
        super().__init__()
        self.logger_ref = logger
        self.save_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(self.save_dir, exist_ok=True)

        self.patience = patience
        self.min_delta = min_delta
        self.check_finite = check_finite
        self.save_weights_only = save_weights_only

        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.mode = mode

        self.best_score = float("inf") if mode == "min" else -float("inf")
        self.wait_count = 0  

    def _is_improved(self, current: float) -> bool:
        if self.check_finite and not torch.isfinite(torch.tensor(current)):
            return False

        if self.mode == "min":
            return current < self.best_score - self.min_delta
        else:  # mode == "max"
            return current > self.best_score + self.min_delta

    def on_validation_end(self, trainer, pl_module):
        monitor = self.logger_ref.monitor_metric
        scores = self.logger_ref.epoch_losses.get(monitor, [])
        if len(scores) == 0:
            return

        epoch, current_score = scores[-1]
        current_score = float(current_score)

        if self._is_improved(current_score):
            self.best_score = current_score
            self.wait_count = 0

            path = os.path.join(self.save_dir, f"model.ckpt")
            if self.save_weights_only:
                torch.save(pl_module.state_dict(), path)
            else:
                # trainer.save_checkpoint(path)
                torch.save(pl_module, path)

            pl_module.print()
            pl_module.print(f"✅ New best model saved (epoch {epoch}, {monitor}: {current_score:.4f})")
            pl_module.print()
        else:
            self.wait_count += 1
            pl_module.print()
            pl_module.print(f"⚠️  No improvement in {monitor} for {self.wait_count} val checks.")
            if self.wait_count >= self.patience:
                pl_module.print()
                pl_module.print(f"⏹️  Early stopping triggered. Best {monitor}: {self.best_score:.4f}")
                trainer.should_stop = True
            pl_module.print()

        # Update plots if needed
        if self.logger_ref.current_epoch % self.logger_ref.plot_frequency == 0:
            self.logger_ref._update_plots()

