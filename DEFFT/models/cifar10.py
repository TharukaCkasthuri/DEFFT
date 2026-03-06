
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
import torch.nn.functional as F
from torch import nn

class CIFAR10Net(nn.Module):
    def __init__(self):
        super().__init__()

        # -------- Block 1 --------
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.gn1   = nn.GroupNorm(4, 32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.gn2   = nn.GroupNorm(4, 32)

        # -------- Block 2 --------
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn3   = nn.GroupNorm(8, 64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.gn4   = nn.GroupNorm(8, 64)

        self.pool = nn.MaxPool2d(2, 2)

        # -------- Classifier --------
        self.fc1 = nn.Linear(64 * 8 * 8, 256)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = F.relu(self.gn2(self.conv2(x)))
        x = self.pool(x)

        x = F.relu(self.gn3(self.conv3(x)))
        x = F.relu(self.gn4(self.conv4(x)))
        x = self.pool(x)

        x = x.view(x.size(0), -1)

        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x