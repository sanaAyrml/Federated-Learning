import torch
from typing import Tuple, Optional, List, Dict
from torch import nn, optim
import time
import copy
from nets.models import DigitModel
import argparse
import numpy as np
import torchvision
import torchvision.transforms as transforms
from utils import data_utils
import torch.nn.utils.weight_norm as weightNorm
import torch.nn.functional as func
from torch.optim import SGD

class BackBone(nn.Module):
    """
    Model for benchmark experiment on Digits.
    """

    def __init__(self, num_classes=10, **kwargs):
        super(BackBone, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5, 1, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 5, 1, 2)
        self.bn3 = nn.BatchNorm2d(128)

        self.fc1 = nn.Linear(6272, 2048)
        self.bn4 = nn.BatchNorm1d(2048)

    #     self.fc2 = nn.Linear(2048, 512)
    #     self.bn5 = nn.BatchNorm1d(512)
    #     self.fc3 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = func.relu(self.bn1(self.conv1(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn2(self.conv2(x)))
        x = func.max_pool2d(x, 2)

        x = func.relu(self.bn3(self.conv3(x)))

        x = x.view(x.shape[0], -1)

        x = self.fc1(x)
        x = self.bn4(x)
        x = func.relu(x)

        #    x = self.fc2(x)
        #    x = self.bn5(x)
        #    x = func.relu(x)

        #    x = self.fc3(x)
        return x


class ImageClassifier(nn.Module):

    def __init__(self, num_classes: int = 10, bottleneck_dim: Optional[int] = 512, **kwargs):
        super(ImageClassifier, self).__init__()
        self.backbone = BackBone()
        self.num_classes = num_classes
        self.bottleneck = nn.Sequential(
            nn.Linear(2048, bottleneck_dim),
            nn.BatchNorm1d(bottleneck_dim),
            nn.ReLU()
        )
        self._features_dim = bottleneck_dim
        self.head = nn.Linear(self._features_dim, num_classes)

    @property
    def features_dim(self) -> int:
        """The dimension of features before the final `head` layer"""
        return self._features_dim

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """"""
        x = self.backbone(x)
        f = self.bottleneck(x)
        predictions = self.head(f)
        return predictions, f

    def get_parameters(self) -> List[Dict]:
        """A parameter list which decides optimization hyper-parameters,
            such as the relative learning rate of each layer
        """
        params = [
            {"params": self.backbone.parameters(), "lr_mult": 0.1},
            {"params": self.bottleneck.parameters(), "lr_mult": 1.},
            {"params": self.head.parameters(), "lr_mult": 1.},
        ]
        return params