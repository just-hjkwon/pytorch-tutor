import functools
import os

from dummy_dataset import DummyDataset


print = functools.partial(print, flush=True)

gpu = "0"
gpu_count = len(gpu.split(','))

learning_rate= 0.001
weight_decay = 0.0001

batch_size = 2
num_workers = 1

load_snapshot = None

use_cuda = True

if use_cuda:
    kwargs = {'num_workers': num_workers, 'pin_memory': True}
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
else:
    kwargs = {}


def main():
    init_epoch = 0
    max_epoch = 20190923

    train_data_set, val_data_set = prepare_data_sets()


def prepare_data_sets():
    train_data_set = DummyDataset(True, 256)
    test_data_set = DummyDataset(False, 256)

    return train_data_set, test_data_set


if __name__ == '__main__':
    main()
