# file to define the training and testing logic for the neural networks

import torch
import torch.nn as nn
import time
from torch.utils.tensorboard import SummaryWriter

def train_network(
    dataloader: torch.utils.data.DataLoader,
    model: nn.Module,
    loss_fn: nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    device: str,
    print_out=True,
):

    # get the full size of training data
    # and the number of batches
    data_size = len(dataloader.dataset)
    num_batches = len(dataloader)

    # set the model to training mode
    # important for batch normalization and dropout layer function
    model.train()

    # initialize epoch loss and number correct
    epoch_loss = 0.0
    correct_model = 0

    # iterate through all the batches of the training data
    for batch, (X, y) in enumerate(dataloader):

        # get the batch size
        batch_size = len(X)

        # transfer the input and the labels to the current device
        X, y = X.to(device), y.to(device)

        # Compute prediction error for the batch
        pred = model(X)
        loss = loss_fn(pred, y)

        # add the current loss to the running total epoch loss
        epoch_loss += loss.item()

        # get the number of correct predictions for the batch
        # and add the number of correct predictions to the running total
        batch_correct_model = (pred.argmax(1) == y).type(torch.float).sum().item()
        correct_model += batch_correct_model

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # print batch loss and number of data processed so far if requested
        if batch % 100 == 0 and print_out:
            loss, current = loss.item(), (batch + 1) * batch_size
            print(
                f"batch loss: {loss:>7f}, data processed: {current:>5d}/{data_size:>5d}"
            )

    # calculate the training accuracy for the epoch
    epoch_accuracy = (correct_model / data_size) * 100

    # calculate the batch average epoch loss
    average_train_loss = epoch_loss / num_batches

    # return the total training loss and epoch accuracy
    # for the epoch to the calling procedure, as these values need to be logged
    # externally
    return epoch_loss, epoch_accuracy, average_train_loss


def test_network(
    dataloader: torch.utils.data.DataLoader, model: nn.Module, loss_fn: nn.modules.loss._Loss, device: str
):

    # get the full size of the test dataset
    # and the number of batches in the dataset
    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    # set the model to testing mode
    # important for batch normalization and dropout layer function
    model.eval()

    # initialize the test loss and the number of correct predictions
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients
    # are computed during test mode and also serves to reduce unnecessary
    # gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():

        # iterate over all data
        for X, y in dataloader:

            # transfer the input and the labels to the current device
            X, y = X.to(device), y.to(device)

            # Compute prediction error for the batch
            pred = model(X)
            loss = loss_fn(pred, y)

            # add the current loss to the running total test loss
            test_loss += loss.item()

            # get the number of correct predictions for the batch and
            # add the number of correct predictions to the running total
            batch_correct_model = (pred.argmax(1) == y).type(torch.float).sum().item()
            correct += batch_correct_model

    # calculate the average test loss and
    # total test accuracy
    average_test_loss = test_loss / num_batches
    test_accuracy = (correct / size) * 100

    # return the total test loss, test accuracy for the epoch,
    # and average test loss to the calling procedure,
    # as these values need to be logged externally
    return test_loss, test_accuracy, average_test_loss


def train_test_network_loop(
    num_epochs: int,
    train_loader: torch.utils.data.DataLoader,
    test_loader: torch.utils.data.DataLoader,
    model: nn.Module,
    loss_fn: nn.modules.loss._Loss,
    optimizer: torch.optim.Optimizer,
    device: str,
    writer: None | SummaryWriter,
    param_file: None | str,
    model_name: None | str,
    print_out=True,
    log_tb=False,
    save_params=False,
):

    # If the user requests to log the results with TensorBoard
    # but doesn't supply a SummaryWriter object, exit with a notification
    if writer is None and log_tb:
        print(
            "If you need to log results to TensorBoard, please supply a SummaryWriter"
        )
        return -1

    # if the user requests to save the parameters of the trained model
    # but does not specify a file name, exit with a notification
    if param_file is None and save_params:
        print(
            "If you need to save the trained network parameters, please supply a file name"
        )
        return -1

    # initialize lists to plot relevant parameters
    train_loss_list = []
    train_accuracy_list = []
    avg_train_loss_list = []
    test_loss_list = []
    test_accuracy_list = []
    avg_test_loss_list = []
    train_duration_list = []
    cumulative_train_duration_list = []
    test_duration_list = []
    cumulative_test_duration_list = []
    
    current_train_duration = 0
    current_test_duration = 0

    # loop over all the training and testing data a set number of times
    for t in range(num_epochs):

        # print header if requested
        if print_out:
            
            if model_name is not None:
                print("\nModel: ", model_name)
            
            print(f"Epoch: {t+1}\n-------------------------------")

        # get training loss and accuracy for the epoch
        epoch_train_start = time.time()
        train_epoch_loss, train_epoch_accuracy, epoch_avg_train_loss = train_network(
            dataloader=train_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            print_out=print_out,
        )
        epoch_train_end = time.time()
        epoch_train_duration = (epoch_train_end - epoch_train_start)
        current_train_duration += epoch_train_duration

        # capture values in lists
        train_loss_list.append(train_epoch_loss)
        train_accuracy_list.append(train_epoch_accuracy)
        avg_train_loss_list.append(epoch_avg_train_loss)
        train_duration_list.append(epoch_train_duration)
        cumulative_train_duration_list.append(current_train_duration)

        # if requested, print the values
        if print_out:
            print("Train epoch loss: ", train_epoch_loss)
            print("Train epoch accuracy: ", train_epoch_accuracy)
            print("Batch average train epoch loss: ", epoch_avg_train_loss)
            print("Epoch training duration, seconds: ", epoch_train_duration)
            print("Cumulative training duration, seconds: ", current_train_duration)

        # if requested, write the values to TensorBoard
        if log_tb:
            writer.add_scalar("Loss/Train/Total", train_epoch_loss, t)
            writer.add_scalar("Accuracy/Train", train_epoch_accuracy, t)
            writer.add_scalar("Loss/Train/Batch_Average", epoch_avg_train_loss, t)
            writer.add_scalar("Duration/Train/Epoch", epoch_train_duration, t)
            writer.add_scalar("Duration/Train/Cumulative", current_train_duration, t)

        # get the total test loss, total accuracy, and average test loss for the epoch
        epoch_test_start = time.time()
        test_epoch_loss, test_epoch_accuracy, epoch_avg_test_loss = test_network(
            dataloader=test_loader, model=model, loss_fn=loss_fn, device=device
        )
        epoch_test_end = time.time()
        epoch_test_duration = (epoch_test_end - epoch_test_start)
        current_test_duration += epoch_test_duration

        # capture values in lists
        test_loss_list.append(test_epoch_loss)
        test_accuracy_list.append(test_epoch_accuracy)
        avg_test_loss_list.append(epoch_avg_test_loss)
        test_duration_list.append(epoch_test_duration)
        cumulative_test_duration_list.append(current_test_duration)

        # if requested, print the values
        if print_out:
            print("Test epoch loss: ", test_epoch_loss)
            print("Test epoch accuracy: ", test_epoch_accuracy)
            print("Batch average test epoch loss: ", epoch_avg_test_loss)
            print("Epoch test duration, seconds: ", epoch_test_duration)
            print("Cumulative test duration, seconds: ", current_test_duration)

        # if requested, write the values to TensorBoard
        if log_tb:
            writer.add_scalar("Loss/Test/Total", test_epoch_loss, t)
            writer.add_scalar("Accuracy/Test", test_epoch_accuracy, t)
            writer.add_scalar("Loss/Test/Batch_Average", epoch_avg_test_loss, t)
            writer.add_scalar("Duration/Test/Epoch", epoch_test_duration, t)
            writer.add_scalar("Duration/Test/Cumulative", current_test_duration, t)

    # if requested, save the parameter values into a file
    if save_params:
        torch.save(model.state_dict(), param_file)

    if log_tb:
        X, y = next(iter(train_loader))
        X, y = X.to(device), y.to(device)
        writer.add_graph(model, X)

    # if requested, print that the loops are done
    if print_out:
        print("Done!")

    return (
        train_loss_list,
        train_accuracy_list,
        avg_train_loss_list,
        train_duration_list,
        cumulative_train_duration_list,
        test_loss_list,
        test_accuracy_list,
        avg_test_loss_list,
        test_duration_list,
        cumulative_test_duration_list
    )