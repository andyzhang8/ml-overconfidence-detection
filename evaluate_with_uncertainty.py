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

def get_uncertainty_model(base_model, method, dataset):
    if method == "mc_dropout":
        return MonteCarloDropout(base_model)
    elif method == "deep_ensemble":
        return DeepEnsemble(model_name, dataset.input_dim, dataset.num_classes)
    elif method == "bayesian_nn":
        return BayesianLinear(dataset.input_dim, dataset.num_classes)
    elif method == "temperature_scaling":
        return TemperatureScaling()
    else:
        raise ValueError(f"Unknown uncertainty estimation method: {method}")

def evaluate_uncertainty(model_name, dataset_name, method, checkpoint_path):
    dataset = StrokeDataset() if dataset_name == "medical" else (
        FinancialDistressDataset() if dataset_name == "financial" else MalwareDataset()
    )

    test_loader = DataLoader(dataset.test_data, batch_size=32, shuffle=False)

    base_model = load_model(model_name, input_dim=dataset.input_dim, num_classes=dataset.num_classes)
    base_model.load_state_dict(torch.load(checkpoint_path))
    model = get_uncertainty_model(base_model, method, dataset)
    model.eval()

    all_probs, all_logits, all_labels = [], [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            logits, uncertainty = model(inputs)
            probs = F.softmax(logits, dim=-1)

            all_logits.append(logits.cpu())
            all_probs.append(probs.cpu())
            all_labels.append(labels.cpu())

    all_probs = torch.cat(all_probs)
    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)

    ece = compute_ece(all_probs, all_labels)
    nll = compute_nll(all_logits, all_labels)
    brier = compute_brier_score(all_probs, all_labels, num_classes=dataset.num_classes)

    print(f"ECE: {ece:.4f}, NLL: {nll:.4f}, Brier Score: {brier:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Model with Uncertainty Estimation")
    parser.add_argument("--model", type=str, required=True, choices=["resnet", "densenet"])
    parser.add_argument("--dataset", type=str, required=True, choices=["medical", "financial", "security"])
    parser.add_argument("--method", type=str, required=True, choices=["mc_dropout", "deep_ensemble", "bayesian_nn", "temperature_scaling"])
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    evaluate_uncertainty(args.model, args.dataset, args.method, args.checkpoint)
