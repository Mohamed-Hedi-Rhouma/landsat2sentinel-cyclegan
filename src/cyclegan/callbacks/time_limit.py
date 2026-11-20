#!/usr/bin/env python
"""Stop before the kaggle session limit: save last.ckpt and end training, counted from this session's start."""

import logging                                   # structured logging
import time                                      # session clock
from pathlib import Path                         # checkpoint path

import pytorch_lightning as pl                   # callback base class

logger = logging.getLogger(__name__)


class SessionTimeLimit(pl.Callback):
    """After max_hours of this session, save last.ckpt at the current step and stop training.

    The clock starts when the callback is built, so every session gets the full budget;
    nothing is stored in the checkpoint, unlike trainer.max_time.
    """

    def __init__(
        self,
        dirpath: str,                            # same folder as the model checkpoints
        max_hours: float = 11.0,                
    ):
        """Start the clock."""
        super().__init__()
        self.path = Path(dirpath) / "last.ckpt"
        self.max_hours = max_hours
        self.start = time.monotonic()

    def hours(self) -> float:
        return (time.monotonic() - self.start) / 3600

    def on_train_batch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule, outputs, batch, batch_idx):
        """Check the clock after every training step."""
        if trainer.should_stop or self.hours() < self.max_hours:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_checkpoint(self.path)
        logger.info("session limit %.1f h reached at step %d: saved %s, stopping", self.max_hours,
                    trainer.global_step, self.path)
        trainer.should_stop = True