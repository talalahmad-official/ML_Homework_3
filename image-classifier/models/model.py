#model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class CIFAR10CNN(nn.Module):
    def __init__(self):
        super(CIFAR10CNN, self).__init__()
        
        # --- Convolutional Layers (Feature Extractors) ---
        
        # Layer 1: Input (3, 32, 32) -> Output (32, 32, 32)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # Shrunk to 16x16
        
        # Layer 2: Input (32, 16, 16) -> Output (64, 16, 16)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # Shrunk to 8x8
        
        # Layer 3: Input (64, 8, 8) -> Output (128, 8, 8)
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # Shrunk to 4x4
        
        # --- Fully Connected Layers (The Classifier) ---
        
        # MATH UPDATE: 128 channels * 4 height * 4 width = 2048
        self.fc1 = nn.Linear(128 * 4 * 4, 256) # Increased neurons to 256
        self.fc2 = nn.Linear(256, 10)          # 10 output classes

    def forward(self, x):
        # Block 1
        x = self.pool1(F.relu(self.conv1(x)))
        
        # Block 2
        x = self.pool2(F.relu(self.conv2(x)))
        
        # Block 3
        x = self.pool3(F.relu(self.conv3(x)))
        
        # Flatten: The output of pool3 is (Batch, 128, 4, 4)
        x = x.view(-1, 128 * 4 * 4)
        
        # Dense Layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        
        return x
