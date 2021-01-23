import argparse
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision.datasets import CIFAR10
from torchvision import transforms
import pytorch_lightning as pl
import numpy as np

from adabelief_pytorch import AdaBelief

from icecream import ic

from lilseb.algebra import Metric
from lilseb.pytorch import *


class BasicCifar10Classifier(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 8, 3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.AvgPool2d(2),

            nn.Conv2d(8, 16, 3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.AvgPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1, padding_mode='replicate'),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, padding_mode='replicate'),
            nn.ReLU(),

            nn.Flatten(),

            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 10),

            nn.Softmax(dim=1),
        )

        self.acc = pl.metrics.classification.Accuracy()

    def forward(self, x):
        prediction = self.model(x)
        return prediction

    def configure_optimizers(self):
        return AdaBelief(
            self.parameters(),
            lr=1e-3,
            eps=1e-8,
            weight_decay=5e-4,
            weight_decouple=False,
            rectify=False)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        z = self.model(x)
        loss = F.cross_entropy(z, y)
        self.log('train_acc', self.acc(z, y))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        z = self.model(x)
        loss = F.cross_entropy(z, y)
        self.log('val_acc', self.acc(z, y))
        self.log('val_loss', loss)


class GACifar10Classifier(pl.LightningModule):
    def __init__(self, metric):
        super().__init__()

        self.metric = metric
        self.model = nn.Sequential(
            SimpleEmbedToGA(metric),

            GPConv2d(metric, 1, 4, 3, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            GPConv2d(metric, 4, 4, 2, stride=2, padding=1, padding_mode='replicate'),
            nn.ReLU(),

            GPConv2d(metric, 4, 8, 3, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            GPConv2d(metric, 8, 8, 2, stride=2, padding=1, padding_mode='replicate'),
            nn.ReLU(),

            GPConv2d(metric, 8, 16, 3, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            GPConv2d(metric, 16, 16, 3, stride=2, padding=1, padding_mode='replicate'),
            nn.ReLU(),

            GPConv2d(metric, 16, 32, 3, padding=1, padding_mode='replicate'),
            nn.ReLU(),
            GPConv2d(metric, 32, 32, 3, padding=1, padding_mode='replicate'),
            nn.ReLU(),

            GAFlatten(metric),

            GPLinear(metric, 800, 256),
            nn.ReLU(),
            GPLinear(metric, 256, 64),
            nn.ReLU(),
            GPLinear(metric, 64, 10),

            SimpleGANormToFeatures(metric),

            nn.Softmax(dim=1),
        )

        self.acc = pl.metrics.classification.Accuracy()

    def forward(self, x):
        prediction = self.model(x)
        return prediction

    def configure_optimizers(self):
        return AdaBelief(
            self.parameters(),
            lr=1e-3,
            eps=1e-8,
            weight_decay=5e-4,
            weight_decouple=False,
            rectify=False)

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        z = self.model(x)
        loss = F.cross_entropy(z, y)
        self.log('train_acc', self.acc(z, y))
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        z = self.model(x)
        loss = F.cross_entropy(z, y)
        self.log('val_acc', self.acc(z, y))
        self.log('val_loss', loss)


parser = argparse.ArgumentParser(description='')
parser.add_argument('--path', default='/data/tvds/cifar-10',
                    help='Path of dataset')
parser.add_argument('--nonga', default=False,
                    action='store_true', help='Train a non-GA model')

args = parser.parse_args()


# data
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
dataset = CIFAR10(args.path, train=True, download=True, transform=transform)
cifar10_train, cifar10_val = random_split(dataset, [45000, 5000])
cifar10_test = CIFAR10(args.path, train=False,
                       download=True, transform=transform)

train_loader = DataLoader(cifar10_train, batch_size=128, num_workers=16)
val_loader = DataLoader(cifar10_val, batch_size=128, num_workers=16)
test_loader = DataLoader(cifar10_test, batch_size=32)

# model
if args.nonga:
    model = BasicCifar10Classifier()
else:
    A = np.array([
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, -1],
        [0, 0, 0, -1, 0]])
    M = Metric(A)
    model = GACifar10Classifier(M)

# training
trainer = pl.Trainer(
    gpus=1,
    terminate_on_nan=True,
    max_epochs=10000)
trainer.fit(model, train_loader, val_loader)
trainer.test(model, test_loader)
