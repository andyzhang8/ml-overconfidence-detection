import torch
import torch.nn as nn
import torch.nn.functional as F

class MonteCarloDropout(nn.Module):
    def __init__(self, model, dropout_prob=0.2, num_samples=10):
        super(MonteCarloDropout, self).__init__()
        self.model = model
        self.dropout_prob = dropout_prob
        self.num_samples = num_samples

        self.enable_dropout(self.model)

    def enable_dropout(self, model):
        for module in model.modules():
            if isinstance(module, nn.Dropout):
                module.train()

    def forward(self, x):
        outputs = torch.stack([self.model(x) for _ in range(self.num_samples)])
        mean_output = outputs.mean(dim=0)  
        uncertainty = outputs.std(dim=0)   
        return mean_output, uncertainty
