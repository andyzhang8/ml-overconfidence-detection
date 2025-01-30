import torch
import numpy as np
import torch.nn.functional as F

def compute_ece(probs, labels, num_bins=10):
    bin_boundaries = torch.linspace(0, 1, num_bins + 1, device=probs.device)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = torch.zeros(1, device=probs.device)
    confidences, predictions = torch.max(probs, dim=1)

    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        mask = (confidences > bin_lower) & (confidences <= bin_upper)
        
        if mask.sum() > 0:  
            bin_conf = confidences[mask].mean()
            bin_acc = (predictions[mask] == labels[mask]).float().mean()  

            ece += (bin_conf - bin_acc).abs() * (mask.float().mean())

    return ece.item()
def compute_nll(logits, labels):
    log_probs = F.log_softmax(logits, dim=-1)
    loss = F.nll_loss(log_probs, labels)
    return loss.item()

def compute_brier_score(probs, labels, num_classes):
    one_hot_labels = F.one_hot(labels, num_classes=num_classes).float()

    if probs.shape[1] != num_classes:
        raise ValueError(f"Shape mismatch: probs has {probs.shape[1]} classes, expected {num_classes}.")

    return torch.mean(torch.sum((probs - one_hot_labels) ** 2, dim=-1)).item()


def compute_aurc(risks, confidences):
    sorted_indices = torch.argsort(confidences, descending=True)
    sorted_risks = risks[sorted_indices]

    coverage = torch.arange(1, len(risks) + 1, dtype=torch.float) / len(risks)
    aurc = torch.trapz(sorted_risks, coverage)
    return aurc.item()
