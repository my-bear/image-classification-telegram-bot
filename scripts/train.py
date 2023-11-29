import os
import json
import argparse

import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets

from image_recognition.tools.pytorchtools import EarlyStopping
from image_recognition.model.NET import ModelMNIST
from image_recognition import transforms


if __name__ == "__main__":

    # Parser for loading the user's path
    parser = argparse.ArgumentParser(description = 'Setting download paths.')

    parser.add_argument(
        '--path_data',
        type = str,
        default = None,
        help = 'the path to loading the dataset'
    )

    parser.add_argument(
        '--path_checkpoint',
        type = str,
        default = None,
        help = 'the path to loading the dataset'
    )
    args = parser.parse_args()

    # Getting custom paths
    with open(os.path.join(os.path.dirname(__file__),\
                        "destination_path.json")) as f:
        destination = json.load(f)

    if args.path_data is not None:
        destination['path_data'] = args.path_data
    if args.path_checkpoint is not None:
        destination['path_checkpoint'] = args.path_checkpoint

    # Set device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Getting the configuration
    with open(os.path.join(os.path.dirname(__file__), \
                            "training_config.json")) as f:
        training_config = json.load(f)

    # Data preparation
    image_datasets = {x: datasets.ImageFolder(os.path.abspath(os.path.join( \
                                            destination['path_data'], x)),
                                            transforms.dict_transforms[x])
                    for x in ['train', 'val']}
    
    dataloaders = {x: DataLoader(image_datasets[x],
                                 batch_size = training_config['batch_size'],
                                 shuffle=True, num_workers=4)
                    for x in ['train', 'val']}
 
    # Initializing the model
    model = ModelMNIST(
        hidden_size = training_config['hidden_size'],
        dropout_prob = training_config['dropout_prob'],
        ).to(device)

    # to track the training loss as the model trains
    train_losses = []

    # to track the validation loss as the model trains
    valid_losses = []

    # to track the average training loss per epoch as the model trains
    avg_train_losses = []

    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = [] 

    # early stopping patience; how long to wait after 
    # last time validation loss improved.
    patience = training_config['patience']

    # initialize the early_stopping object
    early_stopping = EarlyStopping(
        patience = patience,
        verbose = True,
        path = os.path.abspath(destination['path_checkpoint'])
         )

    epochs = training_config['epochs']

    for epoch in range(1, epochs + 1):
        criterion = nn.CrossEntropyLoss()

        optimizer = optim.Adam(
            model.parameters(),
            lr = training_config['lr'],
            weight_decay = training_config['weight_decay']
             )

        # Set model to training mode
        model.train(True)
        for i, data in enumerate(dataloaders['train'], 0):

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # backward + optimize only if in training phase
            loss.backward()
            optimizer.step()

            # record training loss
            train_losses.append(loss.item())

        # eval the model
        correct = 0
        total = 0

        # Set model to evaluate mode
        model.eval()

        with torch.no_grad():
            for i, data in enumerate(dataloaders['val'], 0):

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels).item()
                valid_losses.append(loss)

                predicted = torch.max(outputs.data, 1)[1]
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        train_loss = np.average(train_losses)
        valid_loss = np.average(valid_losses)
        avg_train_losses.append(train_loss)
        avg_valid_losses.append(valid_loss)
        accuracy = correct / total

        epoch_len = len(str(training_config['epochs']))

        # The print a message for output to the console
        print_msg = (f'[{epoch:>{epoch_len}}/{epochs:>{epoch_len}}] ' +
          f'train_loss: {train_loss:.5f} ' +
          f'valid_loss: {valid_loss:.5f} ' +
          f'accuracy: {accuracy:.5f}')
        
        print(print_msg)

        # clear lists to track next epoch
        train_losses.clear()
        valid_losses.clear()
        
        # early_stopping needs the validation loss to check if it has decresed, 
        # and if it has, it will make a checkpoint of the current model
        params = {
          'epoch': epoch,
          'accuracy': accuracy,
          'avg_train_losses': avg_train_losses,
          'avg_valid_losses': avg_valid_losses,
          'optimizer_state_dict': optimizer.state_dict()
            }
        
        early_stopping(valid_loss, model, params)
        
        if early_stopping.early_stop:
          print("Early stopping")
          break
