import os
import sys
import json
import pandas as pd
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

from tabulate import tabulate
from glob import glob
from torchvision import datasets
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), 'src')))
from image_recognition.model.NET import ModelMNIST
from image_recognition import transforms


# Parser for loading the user's path
parser = argparse.ArgumentParser(description = 'Setting download paths.')
parser.add_argument('--path_data', type = str, default = None, help = 'the path to loading the dataset')
parser.add_argument('--model_load', type = str, default = None, help = 'the path to loading the model')
args = parser.parse_args()

if __name__ == "__main__":

    with open(os.path.join(os.path.dirname(__file__), \
                        "destination_path.json")) as f:
        destination = json.load(f)

    if args.path_data is not None:
        destination['path_data'] = args.path_data
    if args.model_load is not None:
        destination['path_model_load'] = args.model_load

    # Getting the configuration
    with open(os.path.join(os.path.dirname(__file__), "training_config.json")) as f:
        training_config = json.load(f)

    # data loader
    image_dataset = datasets.ImageFolder(os.path.abspath(os.path.join( \
                                            destination['path_data'], 'val')),
                                            transforms.dict_transforms['val'])
      
    dataloader = DataLoader(image_dataset, batch_size=training_config['batch_size'],
                                            shuffle = True, num_workers = 4)

    # Initializing the model
    model = ModelMNIST(
        hidden_size = training_config['hidden_size'],
        dropout_prob = training_config['dropout_prob'],
        ).to("cpu")

    # Loss and optimizer #
    criterion = nn.CrossEntropyLoss()

    def evaluate(model, device = 'cpu'):
        # Evaluate on validation set
        val_loss = 0.0
        val_steps = 0
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(dataloader, 0):
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                val_steps += 1

                predicted = torch.max(outputs.data, 1)[1]
                correct += (predicted == labels).sum().item()
                total += labels.size(0)

        val_loss = val_loss / val_steps
        val_accuracy = correct / total
        return round(val_loss, 5), round(val_accuracy, 5)

    df = {'LOSS':[], 'Accuracy':[]}

    # loading the model checkpoint
    checkpoint = torch.load(glob(os.path.abspath(os.path.join( \
                                 destination['path_model_load'], "*.pth")))[-1])
    model.load_state_dict(checkpoint['model_state_dict'])

    loss, accuracy = evaluate(model)
    df['LOSS'].append(loss)
    df['Accuracy'].append(accuracy)

    df = pd.DataFrame(df)
    print(tabulate(df, headers = 'keys', tablefmt = 'psql'))