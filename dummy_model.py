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

    @staticmethod
    def evaluate(prediction, target):
        prediction = nn.Softmax(dim=1)(prediction)
        predicted_id = torch.argmax(prediction, dim=1)

        correct_binaries = torch.where(predicted_id == target, True, False)
        correct_binaries = correct_binaries.cpu().numpy().tolist()

        evaluations = [{'correct': correct_binary}
                       for correct_binary in correct_binaries]

        return evaluations

    @staticmethod
    def compute_error(evaluations):
        correct_count = 0

        for evaluation in evaluations:
            if evaluation['correct'] is True:
                correct_count += 1

        return (correct_count / len(evaluations)) * -1.0

    @staticmethod
    def average_evaluation(evaluations):
        correct_count = 0

        for evaluation in evaluations:
            if evaluation['correct'] is True:
                correct_count += 1

        return {'correct_count': correct_count, 'total_count': len(evaluations)}

    @staticmethod
    def make_evaluation_result_string(evaluation):
        return "Correct: %d / %d" % (evaluation['correct_count'], evaluation['total_count'])
