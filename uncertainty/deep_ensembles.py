import torch
import torch.nn as nn
import torch.optim as optim
from models.model_loader import load_model

class DeepEnsemble:
    def __init__(self, model_name, input_dim, num_classes, num_models=5, learning_rate=0.001):
        self.models = [load_model(model_name, input_dim, num_classes) for _ in range(num_models)]
        self.optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in self.models]
        self.criterion = nn.CrossEntropyLoss()
        self.num_models = num_models

    def train(self, train_loader, epochs=10):
        for model, optimizer in zip(self.models, self.optimizers):
            model.train()
            for epoch in range(epochs):
                for inputs, labels in train_loader:
                    inputs, labels = inputs.to("cuda"), labels.to("cuda")
                    optimizer.zero_grad()
                    outputs = model(inputs)
                    loss = self.criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

    def predict(self, x):
        with torch.no_grad():
            outputs = torch.stack([model(x) for model in self.models])
            mean_output = outputs.mean(dim=0)  
            uncertainty = outputs.std(dim=0)   
        return mean_output, uncertainty
