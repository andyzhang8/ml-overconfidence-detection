import torch
import torch.nn as nn
import torch.nn.functional as F

class DenseLayer(nn.Module):
    def __init__(self, in_features, growth_rate):
        super(DenseLayer, self).__init__()
        self.fc = nn.Linear(in_features, growth_rate)
        self.bn = nn.BatchNorm1d(growth_rate)

    def forward(self, x):
        new_features = F.relu(self.bn(self.fc(x)))
        return torch.cat([x, new_features], dim=1)  

class DenseBlock(nn.Module):
    def __init__(self, in_features, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(DenseLayer(in_features, growth_rate))
            in_features += growth_rate

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class DenseNetTabular(nn.Module):
    def __init__(self, input_dim, num_classes, growth_rate=32, num_blocks=3, num_layers_per_block=2):
        super(DenseNetTabular, self).__init__()
        self.initial_layer = nn.Linear(input_dim, growth_rate)

        adjusted_growth_rate = growth_rate // 2  

        self.blocks = nn.Sequential(
            *[DenseBlock(growth_rate * (i + 1), adjusted_growth_rate, num_layers_per_block) for i in range(num_blocks)]
        )

        final_dim = growth_rate * (num_blocks + 1)  
        self.bottleneck = nn.Linear(final_dim, 64)  
        self.output_layer = nn.Linear(64, num_classes)  

    def forward(self, x):
        x = F.relu(self.initial_layer(x))
        x = self.blocks(x)
        x = F.relu(self.bottleneck(x))  
        return self.output_layer(x)



def densenet(input_dim, num_classes):
    return DenseNetTabular(input_dim=input_dim, num_classes=num_classes)
