import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=0.1):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_std = prior_std

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_log_sigma = nn.Parameter(torch.Tensor(out_features, in_features))

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_log_sigma = nn.Parameter(torch.Tensor(out_features))

        self.reset_parameters()

    def reset_parameters(self):
        self.weight_mu.data.normal_(0, self.prior_std)
        self.weight_log_sigma.data.fill_(-3)
        self.bias_mu.data.normal_(0, self.prior_std)
        self.bias_log_sigma.data.fill_(-3)

    def forward(self, x):
        weight = self.weight_mu + torch.exp(self.weight_log_sigma) * torch.randn_like(self.weight_mu)
        bias = self.bias_mu + torch.exp(self.bias_log_sigma) * torch.randn_like(self.bias_mu)
        return F.linear(x, weight, bias)
