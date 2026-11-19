#!/usr/bin/env python
"""Entry point: build everything from the hydra config and train, resuming from last.ckpt if it exists."""

import logging                                   # structured logging
from pathlib import Path                         # resume checkpoint

import hydra                                     # config framework
import pytorch_lightning as pl                   # trainer
import torch                                     # matmul precision
from hydra.utils import instantiate              # build objects from the config
from omegaconf import DictConfig                 # config type

logger = logging.getLogger(__name__)


def find_resume_checkpoint(log_dir: str) -> str | None:
    """last.ckpt of the run if it exists (kaggle session restarted), else None (fresh run)."""
    last = Path(log_dir) / "checkpoints" / "last.ckpt"
    return str(last) if last.is_file() else None


def instantiate_group(group_cfg) -> list:
    """Every entry of a config group that has a _target_, built into a real object."""
    if not group_cfg:
        return []
    return [instantiate(entry) for entry in group_cfg.values() if entry is not None and "_target_" in entry]


@hydra.main(version_base=None, config_path="../../hydra", config_name="train")
def main(cfg: DictConfig):
    """Build the module, the data, the callbacks, the loggers and the trainer, then fit (and test if asked)."""
    logging.getLogger("cyclegan").setLevel(cfg.loglevel)
    pl.seed_everything(cfg.seed, workers=True)                  # same crops, pools and weights for one seed
    if cfg.get("mat_mul_precision") is not None:                # faster matmuls on recent gpus
        torch.set_float32_matmul_precision(cfg.mat_mul_precision)

    module = instantiate(cfg.training_module.module)            # generators, discriminators, loss
    datamodule = instantiate(cfg.datamodule.loader)             # train, val and test crops
    callbacks = instantiate_group(cfg.callbacks)                # checkpoints, figures, lr monitor
    loggers = instantiate_group(cfg.loggers)                    # tensorboard
    trainer = instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    ckpt_path = find_resume_checkpoint(cfg.log_dir)
    if ckpt_path:
        logger.info("resuming from %s", ckpt_path)
    else:
        logger.info("fresh run in %s", cfg.log_dir)
    trainer.fit(module, datamodule, ckpt_path=ckpt_path)

    if cfg.get("run_test", False):                              # test tiles, once, with the best checkpoint
        trainer.test(module, datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()