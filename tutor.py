import os

import torch
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

    def validate(self, input, target):
        if self.model.training is True:
            self.model.eval()

        if self.use_cuda is True:
            input = input.cuda()
            target = target.cuda()

        prediction = self.model(input)
        loss = self.model.criticize(prediction, target)

        return loss.data, prediction.data

    def save(self, prefix):
        if not os.path.exists(self.snapshot_directory):
            os.mkdir(self.snapshot_directory)

        self.model.eval()

        filename = os.path.join(self.snapshot_directory, "%s.weights" % prefix)
        torch.save(self.model.state_dict(), filename)

        filename = os.path.join(self.snapshot_directory, "%s.optimizer" % prefix)
        torch.save(self.optimizer.state_dict(), filename)

        filename = os.path.join(self.snapshot_directory, "%s.scheduler" % prefix)
        torch.save(self.scheduler.state_dict(), filename)

        train_state = {'best_error': self.best_error, 'epoch': self.epoch}
        filename = os.path.join(self.snapshot_directory, "%s.state" % prefix)
        torch.save(train_state, filename)

    def load(self, prefix):
        self.model.eval()

        filename = os.path.join(self.snapshot_directory, "%s.weights" % prefix)
        self.model.load_state_dict(torch.load(filename))

        filename = os.path.join(self.snapshot_directory, "%s.optimizer" % prefix)
        self.optimizer.load_state_dict(torch.load(filename))

        filename = os.path.join(self.snapshot_directory, "%s.scheduler" % prefix)
        self.scheduler.load_state_dict(torch.load(filename))

        filename = os.path.join(self.snapshot_directory, "%s.state" % prefix)
        train_state = torch.load(filename)

        self.best_error = train_state['best_acer']
        self.epoch = train_state['epoch']

    def start_new_epoch(self):
        self.epoch += 1

    def end_epoch(self, validation_loss):
        self.scheduler.step(validation_loss)

    def get_current_learning_rate(self):
        learning_rate = 0.0

        for param_group in self.optimizer.param_groups:
            learning_rate = param_group['lr']

        return learning_rate

    def make_scheduler_state_string(self):
        state_string = "Best Error: %f, Best loss: %f, Scheduler patience: %d/%d" % (
            self.best_error, self.scheduler.best, self.scheduler.num_bad_epochs, self.scheduler.patience)

        return state_string

    def get_epoch(self):
        return self.epoch
