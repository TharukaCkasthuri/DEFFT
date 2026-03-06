"""
Copyright (C) [2026] Annonymous Author

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

Paper: Hierarchical Knowledge Distillation for Fair Federated Learning
Submitted to: ECML-PKDD 2026 
"""

from torch import nn
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Reduce the number of neurons in the hidden layer
        #self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)  # Output layer with 10 classes
        self.track_layers = {"fc1": self.fc1, "fc2": self.fc2}

    def forward(self, x):
        x = x.view(-1, 28 * 28)  # Flatten the input
        x = self.fc1(x)
        #x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class MNISTLightCNN(nn.Module):
    def __init__(self):
        super(MNISTLightCNN, self).__init__()
        # Feature extractor
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)

        # Classifier
        self.fc1 = nn.Linear(64 * 14 * 14, 128)
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()

        # For hierarchical analysis or layer-wise aggregation if needed
        self.track_layers = {
            "conv1": self.conv1,
            "conv2": self.conv2,
            "fc1": self.fc1,
            "fc2": self.fc2
        }

    def forward(self, x):
        # Input shape: (batch, 1, 28, 28)
        x = self.relu(self.conv1(x))      # -> (batch, 32, 28, 28)
        x = self.pool(self.relu(self.conv2(x)))  # -> (batch, 64, 14, 14)
        x = self.dropout(x)
        x = x.view(-1, 64 * 14 * 14)     # Flatten
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
