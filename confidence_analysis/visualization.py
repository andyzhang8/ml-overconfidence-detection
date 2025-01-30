import torch
import matplotlib.pyplot as plt
import numpy as np

def plot_reliability_diagram(probs, labels, num_bins=10):
    bin_boundaries = torch.linspace(0, 1, num_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    bin_accuracies = []
    bin_confidences = []
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        mask = (probs > bin_lower) & (probs <= bin_upper)
        if mask.any():
            bin_accuracies.append(labels[mask].float().mean().item())
            bin_confidences.append(probs[mask].mean().item())

    plt.figure(figsize=(6, 6))
    plt.plot([0, 1], [0, 1], "k--", label="Perfectly Calibrated")
    plt.scatter(bin_confidences, bin_accuracies, color="red", label="Model Calibration")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability Diagram")
    plt.legend()
    plt.show()

def plot_uncertainty_histogram(uncertainties):
    plt.figure(figsize=(6, 6))
    plt.hist(uncertainties, bins=20, alpha=0.7, color="blue")
    plt.xlabel("Uncertainty")
    plt.ylabel("Frequency")
    plt.title("Uncertainty Histogram")
    plt.show()
