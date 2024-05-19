import timm
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchvision.models import resnet50
from torch.nn import functional as F
from .utils import get_metrics, FocalLoss
from .conv_model import ConvModel
import sys
sys.path.append('../')
import config
from utils.logger import setup

logger = setup("DEBUG")


class BaseModel(pl.LightningModule):
    """
    pretrainedモデルはmodel name にpretrained_XXXとする
    """
    def __init__(self, pm):
        super().__init__()
        self.training_step_outputs = []
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.pm = pm
        if 'pretrained' in pm.model_name:
            self.encoder = self.load_encoder(pm)
            self.fc1 = nn.Linear(1000, pm.fc1_node_n)
            self.fc2 = nn.Linear(pm.fc1_node_n, 1)
        elif pm.mode == 'sv':
            self.model_s = self.create_model(feature_size=pm.feature_size)
            self.model_v = self.create_model(feature_size=pm.feature_size)
            self.fc1 = nn.Linear(pm.feature_size*2, pm.fc1_node_n)
            self.fc2 = nn.Linear(pm.fc1_node_n, pm.num_classes)
        elif pm.mode in ['s', 'v']:
            self.model = self.create_model(feature_size=pm.feature_size)
            self.fc1 = nn.Linear(pm.feature_size, pm.fc1_node_n)
            self.fc2 = nn.Linear(pm.fc1_node_n, pm.num_classes)

        self.criterion = self.get_criterion()

    def create_model(self, feature_size):
        if self.pm.model_name == 'conv':
            return ConvModel(
                input_shape=config.INPUT_SHAPE,
                kernel_size=(self.pm.kernel_height, 3),
                layer_depth=self.pm.layer_depth,
                out_features=feature_size,
            )
        return timm.create_model(
            self.pm.model_name, pretrained=True,
            in_chans=1, num_classes=feature_size
        )

    def get_criterion(self):
        if self.pm.criterion == 'bce':
            return nn.BCEWithLogitsLoss()
        elif self.pm.criterion == 'crossentropy':
            return nn.CrossEntropyLoss()
        elif self.pm.criterion == 'focal':
            return FocalLoss(alpha=self.pm.focal_loss_alpha,
                             gamma=self.pm.focal_loss_gamma)

    def load_encoder(self, pm):
        """load pretrain weights without fc layer weights"""
        encoder = resnet50(pretrained=False)
        state_dict = torch.load(
            config.PRETRAIN_WEGITH_DIR / pm.ckpt
        )['state_dict']
        for key in list(state_dict.keys()):
            if 'model.' in key:
                state_dict.pop(key)
            else:
                state_dict[key.replace('encoder.', '')] = state_dict.pop(key)
        state_dict['fc.weight'] = encoder.fc.weight
        state_dict['fc.bias'] = encoder.fc.bias
        encoder.load_state_dict(state_dict)
        return encoder

    def forward(self, *X):
        if self.pm.mode == 'sv':
            s = X[0]
            v = X[1]
            s = self.model_s(s.float())
            v = self.model_v(v.float())
            cat = torch.cat((s, v), dim=1)
            x = self.fc1(cat)
            x = self.fc2(x)
            return x
        else:
            x = X[0]
            x = self.model(x.float())
            x = self.fc1(x)
            x = self.fc2(x)
            return x

    def training_step(self, batch, batch_idx):
        if self.pm.mode in ['s', 'v']:
            x, y = batch
            y_hat = self(x).softmax(dim=1)
            y = F.one_hot(y.to(torch.int64), num_classes=self.pm.num_classes)
            print('training_step', 'y', y, 'y_hat', y_hat)
            loss = self.criterion(y_hat, y.float())
            self.training_step_outputs.append(
                {
                    'loss': loss, 'y_hat': y_hat,
                    'y': y, 'batch_loss': loss.item() * x.size(0)
                }
            )
            return loss
            print('loss:', loss)
        else:
            s, v, y = batch
            y_hat = self(s, v).softmax(dim=1)
            y = F.one_hot(y.to(torch.int64), num_classes=self.pm.num_classes)
            print('training_step', 'y', y, 'y_hat', y_hat)
            loss = self.criterion(y_hat, y.float())
            print('loss:', loss)
            self.training_step_outputs.append(
                {'loss': loss, 'y_hat': y_hat,
                 'y': y, 'batch_loss': loss * s.size(0)})
            return loss

    def on_training_epoch_end(self):
        y_hat = torch.stack([d['y_hat'] for d in self.train_step_outputs])
        y = torch.stack([d['y'] for d in self.train_step_outputs])
        # y_hat = torch.cat([val['y_hat'] for val in train_step_outputs], dim=0)
        # y = torch.cat([val['y'] for val in train_step_outputs], dim=0)
        epoch_loss = sum(
            [d['batch_loss'] for d in self.train_step_outputs]
        ) / y_hat.size(0)
        self.log_dict(
            {'loss': epoch_loss, **get_metrics(y_hat, y, self.pm.num_classes)},
            on_epoch=True
        )
        self.training_step_outputs.clear()

    def validation_step(self, batch, batch_idx):
        if self.pm.mode in ['s', 'v']:
            x, y = batch
            y_hat = self(x).softmax(dim=1)
            y = F.one_hot(y.to(torch.int64), num_classes=self.pm.num_classes)
            print('validation_step', 'y', y, 'y_hat', y_hat)
            loss = self.criterion(y_hat, y.float())
            print('loss:', loss)
            self.validation_step_outputs.append(
                {
                    'loss': loss, 'y_hat': y_hat,
                    'y': y, 'batch_loss': loss.item() * x.size(0)
                }
            )
        else:
            s, v, y = batch
            y_hat = self(s, v).softmax(dim=1)
            y = F.one_hot(y.to(torch.int64), num_classes=self.pm.num_classes)
            print('validation_step', 'y', y, 'y_hat', y_hat)
            loss = self.criterion(y_hat, y.float())
            print('loss:', loss)
            self.validation_step_outputs.append(
                {
                    'loss': loss, 'y_hat': y_hat,
                    'y': y, 'batch_loss': loss.item() * s.size(0)
                }
            )

    def on_validation_epoch_end(self):
        y_hat = torch.cat([d['y_hat'] for d in self.validation_step_outputs], dim=0)
        y = torch.cat([d['y'] for d in self.validation_step_outputs], dim=0)
        epoch_loss = sum(
            [val['batch_loss'] for val in self.validation_step_outputs]
        ) / y_hat.size(0)
        metrics_dict = \
            {f'val_{k}': v for k, v in get_metrics(y_hat, y, self.pm.num_classes).items()}
        self.log_dict(
            {'val_loss': epoch_loss, **metrics_dict},
            on_epoch=True
        )
        self.validation_step_outputs.clear()

    def test_step(self, batch, batch_idx):
        if self.pm.mode in ['s', 'v']:
            x, y = batch
            y_hat = self(x).softmax(dim=1)
            y = F.one_hot(y.to(torch.int64), num_classes=self.pm.num_classes)
            print('test_step', 'y', y, 'y_hat', y_hat)
            loss = self.criterion(y_hat, y.float())
            print('loss:', loss)
            self.test_step_outputs.append(
                {
                    'loss': loss, 'y_hat': y_hat,
                    'y': y, 'batch_loss': loss.item() * x.size(0)
                }
            )
        else:
            s, v, y = batch
            y_hat = self(s, v).softmax(dim=1)
            y = F.one_hot(y.to(torch.int64), num_classes=self.pm.num_classes)
            print('test_step', 'y', y, 'y_hat', y_hat)
            loss = self.criterion(y_hat, y.float())
            print('loss:', loss)
            self.test_step_outputs.append(
                {
                    'loss': loss, 'y_hat': y_hat,
                    'y': y, 'batch_loss': loss.item() * s.size(0)
                }
            )

    def on_test_epoch_end(self):
        y_hat = torch.cat([d['y_hat'] for d in self.test_step_outputs], dim=0)
        y = torch.cat([d['y'] for d in self.test_step_outputs], dim=0)
        epoch_loss = sum(
            [val['batch_loss'] for val in self.test_step_outputs]
        ) / y_hat.size(0)
        metrics_dict = \
            {f'test_{k}': v for k, v in get_metrics(y_hat, y, self.pm.num_classes).items()}
        self.log_dict(
            {'test_loss': epoch_loss, **metrics_dict},
            on_epoch=True
        )

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), lr=self.pm.lr)
        scheduler = CosineAnnealingLR(opt, T_max=10)
        return [opt], [scheduler]
