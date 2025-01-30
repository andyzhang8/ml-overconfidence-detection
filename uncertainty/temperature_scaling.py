import torch
import torch.nn as nn
import torch.nn.functional as F

class TemperatureScaling(nn.Module):
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits):
        self.temperature.data = self.temperature.data.to(logits.device)  
        return F.softmax(logits / self.temperature, dim=-1)

    def calibrate(self, validation_loader, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        model.eval()
        self.to(device)  
        logits_list, labels_list = [], []

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(device), labels.to(device)  
                logits = model(inputs)
                logits_list.append(logits)
                labels_list.append(labels)

        logits = torch.cat(logits_list).to(device)  
        labels = torch.cat(labels_list).to(device)

        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def closure():
            optimizer.zero_grad()
            loss = F.cross_entropy(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(closure)

