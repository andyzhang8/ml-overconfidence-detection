import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.model_loader import load_model
from uncertainty.monte_carlo_dropout import MonteCarloDropout
from uncertainty.deep_ensembles import DeepEnsemble
from uncertainty.temperature_scaling import TemperatureScaling
from uncertainty.bayesians_nn import BayesianLinear
from data.medical_loader import StrokeDataset
from data.financial_loader import FinancialDistressDataset
from data.security_loader import MalwareDataset
from confidence_analysis.risk_metrics import compute_ece, compute_nll, compute_brier_score
import os

def get_uncertainty_model(base_model, method, dataset, checkpoint_path):
    print(f"init {method} model...", flush=True)

    if method == "mc_dropout":
        return MonteCarloDropout(base_model)
    elif method == "deep_ensemble":
        return load_deep_ensemble(dataset, checkpoint_path)
    elif method == "bayesian_nn":
        return BayesianLinear(dataset.input_dim, dataset.num_classes)
    elif method == "temperature_scaling":
        return TemperatureScaling()
    else:
        raise ValueError(f"dont know uncertainty estimation method: {method}")

def load_deep_ensemble(dataset, checkpoint_path):
    print(f"Loading Deep Ensemble from {checkpoint_path}...", flush=True)
    checkpoint = torch.load(checkpoint_path, map_location="cuda" if torch.cuda.is_available() else "cpu")

    models = []
    for i in range(5): 
        model = load_model("densenet", input_dim=dataset.input_dim, num_classes=dataset.num_classes)
        model.load_state_dict(checkpoint[f"model_{i}"])
        models.append(model)

    return DeepEnsemble(models=models)

def evaluate_uncertainty(model_name, dataset_name, method, checkpoint_path):
    print(f"Evaluating {method} for {model_name} on {dataset_name}...", flush=True)

    dataset = StrokeDataset() if dataset_name == "medical" else (
        FinancialDistressDataset() if dataset_name == "financial" else MalwareDataset()
    )
    test_loader = DataLoader(dataset.test_data, batch_size=32, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model = load_model(model_name, input_dim=dataset.input_dim, num_classes=dataset.num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        print("Extracting model state_dict...", flush=True)
        checkpoint = checkpoint["model"]

    base_model.load_state_dict(checkpoint, strict=False)
    base_model.to(device)

    model = get_uncertainty_model(base_model, method, dataset, checkpoint_path)
    model.to(device)
    model.eval()

    print("Running inference...", flush=True)

    all_probs, all_logits, all_labels, all_uncertainties, overconfident_preds = [], [], [], [], []

    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(test_loader):
            print(f"Processing batch {batch_idx + 1}/{len(test_loader)}...", flush=True)
            inputs, labels = inputs.to(device), labels.to(device)

            output = model(inputs)

            if isinstance(output, tuple) and len(output) == 2:
                logits, uncertainty = output
            else:
                logits = output 
                uncertainty = torch.zeros_like(logits)  

            probs = F.softmax(logits, dim=-1)
            max_probs, preds = torch.max(probs, dim=1)
            uncertainty = uncertainty.mean(dim=1) if uncertainty.ndim > 1 else uncertainty

            confidence_threshold = 0.9
            uncertainty_threshold = uncertainty.mean().item() + uncertainty.std().item()  # Adaptive threshold

            for i in range(len(labels)):
                if max_probs[i] > confidence_threshold and uncertainty[i] > uncertainty_threshold:
                    overconfident_preds.append((batch_idx * len(labels) + i, preds[i].item(), max_probs[i].item(), uncertainty[i].item()))

            all_logits.append(logits.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())
            all_uncertainties.append(uncertainty.cpu())

    all_probs = torch.cat(all_probs)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    all_uncertainties = torch.cat(all_uncertainties)

    print("Computing evaluation metrics...", flush=True)
    ece = compute_ece(all_probs, all_labels)
    nll = compute_nll(all_logits, all_labels)
    brier = compute_brier_score(all_probs, all_labels, num_classes=dataset.num_classes)

    print("\nEvaluation Results:")
    print(f"ECE: {ece:.4f}", flush=True)
    print(f"NLL: {nll:.4f}", flush=True)
    print(f"Brier Score: {brier:.4f}", flush=True)

    os.makedirs("logs", exist_ok=True)

    log_file = f"logs/eval_{model_name}_{dataset_name}_{method}.txt"
    log_file = log_file.replace(" ", "_").lower()

    with open(log_file, "w") as f:
        f.write(f"Evaluation Results for {model_name} on {dataset_name} using {method}:\n")
        f.write(f"ECE: {ece:.4f}\n")
        f.write(f"NLL: {nll:.4f}\n")
        f.write(f"Brier Score: {brier:.4f}\n")

    overconfident_log_file = f"logs/overconfidence_{model_name}_{dataset_name}_{method}.txt"
    with open(overconfident_log_file, "w") as f:
        f.write(f"Overconfidence Predictions for {model_name} on {dataset_name} using {method}:\n\n")
        f.write(f"{'Sample':<10} {'Pred':<10} {'Confidence':<15} {'Uncertainty':<15}\n")
        f.write("=" * 50 + "\n")
        for sample, pred, confidence, uncertainty in overconfident_preds:
            f.write(f"{sample:<10} {pred:<10} {confidence:<15.4f} {uncertainty:<15.4f}\n")

    print(f"\nEvaluation results saved to {log_file}", flush=True)
    print(f"Overconfidence results saved to {overconfident_log_file}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Model with Uncertainty Estimation")
    parser.add_argument("--model", type=str, required=True, choices=["resnet", "densenet"])
    parser.add_argument("--dataset", type=str, required=True, choices=["medical", "financial", "security"])
    parser.add_argument("--method", type=str, required=True, choices=["mc_dropout", "deep_ensemble", "bayesian_nn", "temperature_scaling"])
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    evaluate_uncertainty(args.model, args.dataset, args.method, args.checkpoint)
