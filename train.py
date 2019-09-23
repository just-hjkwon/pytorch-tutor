import functools


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


if __name__ == '__main__':
    main()
