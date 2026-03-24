import torch.nn as nn
import torch

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()


class Model(nn.Module):
    def __init__(self, dim, num):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, 384), nn.BatchNorm1d(384), nn.ReLU(), nn.Dropout(0.3), nn.Linear(384, 256),
                                 nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(),
                                 nn.Dropout(0.2), nn.Linear(128, num))

    def forward(self, x):
        return self.net(x)