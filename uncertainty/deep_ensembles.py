import torch
import torch.nn as nn
import torch.optim as optim
from models.model_loader import load_model

class DeepEnsemble:
    def __init__(self, models=None, model_name=None, input_dim=None, num_classes=None, num_models=5, learning_rate=0.001):
        if models:
            self.models = models 
        else:
            if not model_name or not input_dim or not num_classes:
                raise ValueError("Must provide model_name, input_dim, and num_classes when initializing from scratch.")
            self.models = [load_model(model_name, input_dim, num_classes) for _ in range(num_models)]

        self.optimizers = [optim.Adam(model.parameters(), lr=learning_rate) for model in self.models]
        self.criterion = nn.CrossEntropyLoss()
        self.num_models = len(self.models)

    def to(self, device):
        for model in self.models:
            model.to(device)

    def train(self):
        for model in self.models:
            model.train()

    def eval(self):
        for model in self.models:
            model.eval()

    def predict(self, x):
        with torch.no_grad():
            outputs = torch.stack([model(x) for model in self.models])
            mean_output = outputs.mean(dim=0)  
            uncertainty = outputs.std(dim=0)   
        return mean_output, uncertainty
