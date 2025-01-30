import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.bn1 = nn.BatchNorm1d(hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.bn2 = nn.BatchNorm1d(in_features)
        self.shortcut = nn.Identity()  

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        x += residual 
        return F.relu(x)

class ResNetTabular(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=128, num_blocks=4):
        super(ResNetTabular, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.bn_input = nn.BatchNorm1d(hidden_dim)

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)]
        )

        self.output_layer = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = F.relu(self.bn_input(self.input_layer(x)))
        x = self.res_blocks(x)
        return self.output_layer(x)

def resnet(input_dim, num_classes):
    return ResNetTabular(input_dim=input_dim, num_classes=num_classes)
