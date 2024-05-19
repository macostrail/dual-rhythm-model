import datetime
import torch
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
import wandb

from utils import utils
from utils.validate import validate_dataset
import config as cfg
from models.base_model import BaseModel
from dataset import EcgDataset
import ecgaugmentor as aug

SEED = 1231
utils.seed_everything(SEED)
generator = torch.Generator()
generator.manual_seed(SEED)
torch.use_deterministic_algorithms(False)


def train(pm):
    if cfg.DATAFRAME.suffix in ['.xlsx', '.xls']:
        df = pd.read_excel(cfg.DATAFRAME, engine='openpyxl')
    elif cfg.DATAFRAME.suffix == '.csv':
        try:
            df = pd.read_csv(cfg.DATAFRAME, encoding='utf-8')
        except:
            df = pd.read_csv(cfg.DATAFRAME, encoding='cp932')
    validate_dataset()
    df = df[~df[pm.target_col].isna()]
    if pm.target_col == 'Orig5':
        df['Orig5'] = df['Orig5'].astype(int) - 1
    tv_df, test_df = train_test_split(df, train_size=0.8, shuffle=True, random_state=SEED)
    train_df, valid_df = train_test_split(tv_df, train_size=0.75, shuffle=True, random_state=SEED)

    transform = aug.Compose([
        aug.BaseLiner(),
        aug.LowHighPass(lowpass_range=cfg.SOFT_LOWPASS_RANGE,
                        highpass_range=cfg.SOFT_HIGHPASS_RANGE),
        aug.StrechTime(cfg.SOFT_STRECHTIME_RANGE),
        aug.StrechVoltage(cfg.SOFT_STRECHVOLTAGE_RANGE),
        aug.RandomNoise(low_fq_hz_max=cfg.SOFT_LOW_FQ_HZ_MAX,
                        low_fq_mv_max=cfg.SOFT_LOW_FQ_MV_MAX,
                        high_fq_hz_max=cfg.SOFT_HIGH_FQ_HZ_MAX,
                        high_fq_mv_max=cfg.SOFT_HIGH_FQ_MV_MAX),
        aug.TimeShift(cfg.SOFT_TIMESHIFT_RANGE)
    ])

    train_dataset = EcgDataset(
        df=train_df,
        pm=pm,
        transform=transform if pm.augmentation else None
    )
    valid_dataset = EcgDataset(
        df=valid_df,
        pm=pm,
        transform=None
    )
    test_dataset = EcgDataset(
        df=test_df,
        pm=pm,
        transform=None
    )
    model = BaseModel(pm)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=pm.batch_size,
        shuffle=True,
        num_workers=pm.num_workers,
        generator=generator
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=pm.batch_size,
        shuffle=False,
        num_workers=pm.num_workers,
        generator=generator
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=pm.batch_size,
        shuffle=False,
        num_workers=pm.num_workers,
        generator=generator
    )

    wandb_logger = WandbLogger(project=cfg.PROJECT)
    now_time = datetime.datetime.now(datetime.timezone.utc).strftime('%Y/%m/%d %H:%M:%S')
    dirpath = '../ckpt/' + now_time

    callbacks = []
    checkpoint_callback = ModelCheckpoint(
            dirpath=dirpath,
            save_top_k=1,
            monitor=pm.checkpoint_monitor,
            mode=pm.checkpoint_monitor_mode
        )
    callbacks.append(checkpoint_callback)

    if pm.earlystopping:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss", mode="min", check_finite=True,
                patience=pm.earlystopping_patience
            )
        )

    trainer = Trainer(
        accelerator="gpu",
        # strategy="ddp",
        precision=16,
        check_val_every_n_epoch=1,
        log_every_n_steps=1,
        min_epochs=pm.min_epochs,
        max_epochs=pm.max_epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_dataloader, valid_dataloader)
    trainer.test(ckpt_path=checkpoint_callback.best_model_path,
                 dataloaders=test_dataloader)


def test_run():
    @dataclass
    class Parameters:
        batch_size: int = 64
        num_classes: int = 2
        input_size = (224, 224, 3)
        epochs: int = 10
        lr: float = 1e-3
        model_name: str = 'eca_nfnet_l0'
        epo_rough: int = 30
        min_epochs: int = 5
        max_epochs: int = 20
        criterion: str = 'bce'
        threshold: int = 15
        num_workers: int = 20
        target_col: str = 'Orig2'
        mode: str = 'sv'
        input_shape = (12, 400)
        in_dims = 3
    pm = Parameters()
    train(pm)


if __name__ == '__main__':
    wandb.init()
    pm = wandb.config
    train(pm)
