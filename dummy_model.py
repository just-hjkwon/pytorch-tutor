import torch
import torch.nn as nn


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()

    def forward(self, input):
        return torch.sum(input * 2.0, dim=(1, 2, 3))
