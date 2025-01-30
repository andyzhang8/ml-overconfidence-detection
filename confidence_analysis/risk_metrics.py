import torch
import numpy as np
import torch.nn.functional as F

def compute_ece(probs, labels, num_bins=10):
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = torch.zeros(1, device=probs.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        mask = (probs > bin_lower) & (probs <= bin_upper)
        bin_prob = probs[mask]
        bin_acc = labels[mask].float()

        if bin_prob.numel() > 0:
            avg_conf = bin_prob.mean()
            avg_acc = bin_acc.mean()
            ece += (avg_conf - avg_acc).abs() * (bin_prob.numel() / probs.numel())

    return ece.item()

def compute_nll(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    loss = F.nll_loss(log_probs, labels)
    return loss.item()

def compute_brier_score(probs, labels, num_classes):
    one_hot_labels = F.one_hot(labels, num_classes=num_classes).float()
    return torch.mean(torch.sum((probs - one_hot_labels) ** 2, dim=-1)).item()

def compute_aurc(risks, confidences):
    sorted_indices = torch.argsort(confidences, descending=True)
    sorted_risks = risks[sorted_indices]

    coverage = torch.arange(1, len(risks) + 1, dtype=torch.float) / len(risks)
    aurc = torch.trapz(sorted_risks, coverage)
    return aurc.item()
