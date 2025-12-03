
import torch
import torch.nn.functional as F
from torch import nn

''''
class CIFAR10Net(nn.Module):
    def __init__(self):
        super(CIFAR10Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(4, 32)
        #self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(8, 64)
        #self.bn2 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(64 * 8 * 8, 128)
        self.dropout = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        #x = self.pool(F.relu(self.gn1(self.conv1(x))))  # [batch_size, 32, 16, 16]
        #x = self.pool(F.relu(self.gn2(self.conv2(x))))  # [batch_size, 64, 8, 8]
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))


        x = x.view(-1, 64 * 8 * 8)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
'''

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



# ---------- Basic Residual Block ----------
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, groups=8):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.gn2 = nn.GroupNorm(groups, out_channels)

        # Shortcut if needed
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(groups, out_channels)
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


# ---------- ResNet-18 for CIFAR-10 ----------
class CIFAR10ResNet18(nn.Module):
    def __init__(self, num_classes=10, groups=8):
        super().__init__()

        self.in_channels = 64
        # No ImageNet-style stem; CIFAR doesn't need it
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(groups, 64)
        # Layers: 2 blocks each (ResNet18)
        self.layer1 = self._make_layer(64, 2, stride=1, groups=groups)
        self.layer2 = self._make_layer(128, 2, stride=2, groups=groups)
        self.layer3 = self._make_layer(256, 2, stride=2, groups=groups)
        self.layer4 = self._make_layer(512, 2, stride=2, groups=groups)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride, groups):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, groups))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1, groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)





import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Basic Residual Block (Corrected Lite Version) ----------
class BasicBlockLite(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, groups=8):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(groups, out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.gn2 = nn.GroupNorm(groups, out_channels)

        # Shortcut: RESTORE GroupNorm for stability
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.GroupNorm(groups, out_channels)   # FIXED
            )

    def forward(self, x):
        out = F.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


# ---------- Lightweight but Stable FL-ResNet18 ----------
class CIFAR10ResNet18Lite(nn.Module):
    def __init__(self, num_classes=10, groups=8):
        super().__init__()

        # Start smaller → stable but efficient
        self.in_channels = 32
        self.conv1 = nn.Conv2d(
            3, 32, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.gn1 = nn.GroupNorm(groups, 32)

        # Lighter ResNet Layout
        self.layer1 = self._make_layer(32,  2, stride=1, groups=groups)
        self.layer2 = self._make_layer(64,  2, stride=2, groups=groups)
        self.layer3 = self._make_layer(128, 2, stride=2, groups=groups)
        self.layer4 = self._make_layer(256, 2, stride=2, groups=groups)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def _make_layer(self, out_channels, blocks, stride, groups):
        layers = []
        layers.append(BasicBlockLite(self.in_channels, out_channels, stride, groups))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(BasicBlockLite(out_channels, out_channels, 1, groups))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.gn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

