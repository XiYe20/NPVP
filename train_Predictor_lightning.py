import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks import Callback
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.plugins import PrecisionPlugin


import torch
import torch.nn as nn
import torch.nn.functional as F

from models import LitAE, Predictor, Discriminator, ResnetEncoder, ResnetDecoder, LitPredictor
from models import L1Loss, Div_KL, GANLoss
from utils import LitDataModule, VisCallbackPredictor, save_code_cfg

import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg : DictConfig) -> None:
    #save the code and config
    save_code_cfg(cfg, cfg.Predictor.ckpt_save_dir)

    pl.seed_everything(cfg.Env.rand_seed, workers=True)
    #init model and dataloader
    data_module = LitDataModule(cfg)
    predictor = LitPredictor(cfg)

    #init logger and all callbacks
    checkpoint_callback = ModelCheckpoint(dirpath=cfg.Predictor.ckpt_save_dir, every_n_epochs = cfg.Predictor.log_per_epochs,
                                          save_top_k= cfg.Predictor.epochs, monitor = 'loss_val', filename= "Predictor-{epoch:02d}")
    if cfg.Env.visual_callback:
        callbacks = [VisCallbackPredictor(), checkpoint_callback]
    else:
        callbacks = [checkpoint_callback]
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=cfg.Predictor.tensorboard_save_dir)
    trainer = pl.Trainer(accelerator="gpu", devices=cfg.Env.world_size,
                         max_epochs=cfg.Predictor.epochs, enable_progress_bar=True, sync_batchnorm=True,
                         callbacks = callbacks, logger=tb_logger, strategy = cfg.Env.strategy)
    if cfg.Predictor.init_det_ckpt_for_vae is not None and cfg.Predictor.resume_ckpt is None:
        predictor = predictor.load_from_checkpoint(cfg = cfg, checkpoint_path=cfg.Predictor.init_det_ckpt_for_vae,
                                                   strict = False)

        trainer.fit(predictor, data_module)
    else:
        trainer.fit(predictor, data_module, ckpt_path=cfg.Predictor.resume_ckpt)

if __name__ == '__main__':
    main()