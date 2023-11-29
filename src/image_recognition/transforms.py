import torch
import torch.nn as nn
from torchvision.transforms import v2


# Data augmentation and normalization for training
# Just normalization for validation
dict_transforms = {
    'train': nn.Sequential(
        v2.RandomRotation([-5, 5]),
        v2.PILToTensor(),
        v2.Grayscale(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean = [0.5], std = [0.5])
    ),
    'val': nn.Sequential(
        v2.PILToTensor(),
        v2.Grayscale(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean = [0.5], std = [0.5])
    ),
}
