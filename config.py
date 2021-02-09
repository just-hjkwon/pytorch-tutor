import logging
import os

from torch.utils.tensorboard import SummaryWriter

from dummy_dataset import DummyDataset
from dummy_model import DummyModel
from tutor import Tutor

tensorboard_log_directory = "./logs"

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


def create_logger():
    logger = logging.getLogger('tutor')
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s %(levelname)s] %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)

    return logger


def create_tensorboard_writer(purge_step=None):
    current_directory_name = os.path.split(os.path.dirname(os.path.realpath(__file__)))[-1]
    log_dir = os.path.join(tensorboard_log_directory, current_directory_name)
    writer = SummaryWriter(log_dir=log_dir, flush_secs=10, purge_step=purge_step)

    return writer


def write_extra_tensorboard_log(writer: SummaryWriter, epoch: int, tutor: Tutor):
    pass
