import os

from dummy_model import DummyModel
from dummy_dataset import DummyDataset

gpu = "0"
gpu_count = len(gpu.split(','))

max_epoch = 20210209

learning_rate = 0.001
weight_decay = 0.0001

batch_size = 2
num_workers = 1

load_snapshot = None

use_cuda = True

if use_cuda:
    cuda_kwargs = {'num_workers': num_workers, 'pin_memory': True}
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
else:
    cuda_kwargs = {}


def create_model():
    model = DummyModel()

    if use_cuda is True:
        if gpu_count > 1:
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = model.cuda()

    return model


def create_data_sets():
    train_data_set = DummyDataset(True, 256)
    test_data_set = DummyDataset(False, 256)

    return train_data_set, test_data_set
