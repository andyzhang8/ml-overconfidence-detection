import torch
import torch.nn as nn
import torch.nn.functional as F

class TemperatureScaling(nn.Module):
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        return F.softmax(logits / self.temperature, dim=-1)

    def calibrate(self, validation_loader, model):
        model.eval()
        logits_list, labels_list = [], []
        with torch.no_grad():
            for inputs, labels in validation_loader:
                logits = model(inputs)
                logits_list.append(logits)
                labels_list.append(labels)

        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / self.temperature, labels)
            loss.backward()
            return loss
        optimizer.step(closure)
