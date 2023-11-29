import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelMNIST(nn.Module):
    def __init__(self, hidden_size, dropout_prob = 0.5):
        super(ModelMNIST, self).__init__()
        self.num_classes = 10
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(dropout_prob/2)
        self.dropout2 = nn.Dropout(dropout_prob)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(hidden_size, self.num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    