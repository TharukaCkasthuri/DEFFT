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

import torch
from torch import nn
import torch.nn.functional as F

class FEMNISTNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # ---- Conv blocks ----
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # GroupNorm > BatchNorm for small / federated batches
        self.gn1 = nn.GroupNorm(4, 32)
        self.gn2 = nn.GroupNorm(8, 64)
        self.gn3 = nn.GroupNorm(16, 128)

        # ---- Classifier ----
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(p=0.3)

        self.track_layers = {
            "conv1": self.conv1,
            "conv2": self.conv2,
            "conv3": self.conv3,
            "fc1": self.fc1,
            "fc2": self.fc2,
        }

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)

        x = F.relu(self.gn1(self.conv1(x)))
        x = F.max_pool2d(x, 2)      # 14×14

        x = F.relu(self.gn2(self.conv2(x)))
        x = F.max_pool2d(x, 2)      # 7×7

        x = F.relu(self.gn3(self.conv3(x)))
        x = F.max_pool2d(x, 2)      # 3×3

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x

    def process_x(self, raw_x_batch):
        x = torch.tensor(raw_x_batch, dtype=torch.float32)
        return (x - 0.5) / 0.5   # normalize

    def process_y(self, raw_y_batch):
        return torch.tensor(raw_y_batch, dtype=torch.long)