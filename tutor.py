import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler

from torch.autograd import Variable


class Tutor:
    def __init__(self, model, learning_rate=0.05, weight_decay=0.0005, use_cuda=True):
        assert(issubclass(model, nn.Module))

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.use_cuda = use_cuda

        target_parameters = self.make_optimizing_target_parameters()

        self.optimizer = optim.Adam(target_parameters, lr=learning_rate, weight_decay=weight_decay)
        self.scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=(2.0/3.0), patience=15)

        self.snapshot_directory = "snapshots"

        self.best_error = 1.0
        self.epoch = 0

    def make_optimizing_target_parameters(self):
        target_parameters = list()

        named_parameters = self.model.named_parameters()

        for name, param in named_parameters:
            if name[-9:] == "conv.bias":
                target_parameters.append({"params": param, 'lr': self.learning_rate * 2.0, 'weight_decay': 0.0})
            else:
                target_parameters.append({"params": param, 'lr': self.learning_rate, 'weight_decay': weight_decay})

        return target_parameters

    def train(self, input, target):
        if self.model.training is not True:
            self.model.train()

        if self.use_cuda is True:
            input = input.cuda()
            target = target.cuda()

        input = Variable(input)
        target = Variable(target)

        self.optimizer.zero_grad()

        prediction = self.model(input)

        loss = self.model.criticize(prediction, target)
        loss.backward()

        self.optimizer.step()

        return loss.data
