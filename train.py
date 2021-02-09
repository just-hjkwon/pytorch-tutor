import time

import torch
import tqdm

import config
from tutor import Tutor

log = config.create_logger()


def main():
    init_epoch = 0
    max_epoch = config.max_epoch

    train_data_set, val_data_set = config.create_data_sets()
    model = config.create_model()

    tutor = Tutor(model=model, learning_rate=config.learning_rate, weight_decay=config.weight_decay)

    load_snapshot = config.load_snapshot

    if load_snapshot is not None:
        tutor.load(load_snapshot)
        validate(tutor, val_data_set)

        init_epoch = tutor.get_epoch() + 1

    writer = config.create_tensorboard_writer(purge_step=init_epoch)

    for epoch in range(init_epoch, max_epoch):
        tutor.set_epoch(epoch)

        train_loss = train_a_epoch(tutor, train_data_set)
        writer.add_scalar("Learning/Train loss", train_loss, epoch)

        validation_loss, error = validate(tutor, val_data_set)
        writer.add_scalar("Learning/Validation loss", validation_loss, epoch)

        current_learning_rate = tutor.get_current_learning_rate()
        writer.add_scalar("Learning/Learning rate", validation_loss, epoch)

        tutor.end_epoch(validation_loss)

        if error <= tutor.best_error:
            tutor.best_error = error
            tutor.save('best')
            tutor.save('best_at_epoch_%04d' % epoch)

            log_string = 'Save snapshot of best ACER (%.4f)' % (tutor.best_error)
            log.info(log_string)

        tutor.save('latest')


def train_a_epoch(tutor, data_set):
    assert (isinstance(tutor, Tutor))

    data_loader = torch.utils.data.DataLoader(data_set, batch_size=config.batch_size, shuffle=False,
                                              **config.cuda_kwargs)

    current_epoch = tutor.get_epoch()
    current_learning_rate = tutor.get_current_learning_rate()

    scheduler_string = tutor.make_scheduler_state_string()

    log_string = 'Train | Epoch %d, learning rate: %f, %s' % (
        current_epoch, current_learning_rate, scheduler_string)

    log.info(log_string)

    epoch_loss = 0.0
    trained_count = 0

    description = get_time_string() + " TQDM]"
    epoch_bar = tqdm.tqdm(data_loader, desc=description)

    for batch_idx, (data, target) in enumerate(epoch_bar):
        loss = tutor.train(data, target)

        trained_count += len(data)
        epoch_loss += loss * len(data)

        average_loss = epoch_loss / trained_count

        time_string = get_time_string()
        description = "%s TQDM] Train | Epoch %d, Average train loss: %f (batch: %f)" % (
            time_string, current_epoch, average_loss, loss)
        epoch_bar.set_description(description)

    average_loss = epoch_loss / trained_count

    description = "Train | Epoch %d, Average train loss: %f" % (current_epoch, average_loss)
    log.info(description)

    return average_loss


def validate(tutor, data_set):
    assert (isinstance(tutor, Tutor))

    data_loader = torch.utils.data.DataLoader(data_set, batch_size=config.batch_size, shuffle=False,
                                              **config.cuda_kwargs)

    current_epoch = tutor.get_epoch()
    current_learning_rate = tutor.get_current_learning_rate()

    log_string = 'Valid.| Epoch %d, learning rate: %f' % (current_epoch, current_learning_rate)
    log.info(log_string)

    validation_loss = 0.0
    validated_count = 0

    description = get_time_string() + " TQDM]"
    epoch_bar = tqdm.tqdm(data_loader, desc=description)

    for batch_idx, (input, target) in enumerate(epoch_bar):
        loss, output = tutor.validate(input, target)

        validated_count += len(input)
        validation_loss += loss * len(input)
        average_loss = validation_loss / validated_count

        time_string = get_time_string()
        description = "%s TQDM] Valid.| Epoch %d, Average validation loss: %f (batch: %f)" % (
            time_string, current_epoch, average_loss, loss)
        epoch_bar.set_description(description)

    average_loss = validation_loss / validated_count

    description = "Valid.| Epoch %d, Average validation loss: %f" % (current_epoch, average_loss)
    log.info(description)

    error = validation_loss

    return average_loss, error


def get_time_string():
    string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    return string


if __name__ == '__main__':
    main()
