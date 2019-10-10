import torch
import torch.nn as nn


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.loss = self.create_loss()

    def forward(self, input):
        return torch.sum(input * 2.0, dim=(1, 2, 3))

    def create_loss(self):
        return nn.CrossEntropyLoss()

    def criticize(self, prediction, target):
        loss = self.loss(prediction, target)
        return loss
