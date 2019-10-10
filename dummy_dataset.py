import torch
from torch.utils.data import Dataset

import numpy as np


class DummyDataset(Dataset):
    def __init__(self, is_train_mode, batch_size):
        self.is_train_mode = is_train_mode
        self.batch_size = batch_size

    def __len__(self):
        return 1024

    def __getitem__(self, item):
        input = np.random.rand(3, 100, 100).astype(dtype=np.float32)
        input = torch.from_numpy(input)

        label = np.random.random_integers(0, 2)

        return input, label
