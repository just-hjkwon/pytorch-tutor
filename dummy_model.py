import torch
import torch.nn as nn


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=(100, 100), padding=0)

        self.loss = self.create_loss()

    def forward(self, input):
        output = self.conv1(input)
        output = output.squeeze()
        return output

    def create_loss(self):
        return nn.CrossEntropyLoss()

    def criticize(self, prediction, target):
        loss = self.loss(prediction, target)
        return loss
