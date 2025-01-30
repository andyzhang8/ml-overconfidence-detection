import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.model_loader import load_model
from uncertainty.monte_carlo_dropout import MonteCarloDropout
from uncertainty.deep_ensembles import DeepEnsemble
from uncertainty.temperature_scaling import TemperatureScaling
from data.medical_loader import StrokeDataset
from data.financial_loader import FinancialDistressDataset
from data.security_loader import MalwareDataset
from confidence_analysis.risk_metrics import compute_ece, compute_nll, compute_brier_score
import os

def compare_uncertainty_methods(model_name, dataset_name, checkpoint_path):
    print(f"compare uncertainty for {model_name} on {dataset_name}...", flush=True)

    dataset = StrokeDataset() if dataset_name == "medical" else (
        FinancialDistressDataset() if dataset_name == "financial" else MalwareDataset()
    )

    test_loader = DataLoader(dataset.test_data, batch_size=32, shuffle=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model = load_model(model_name, input_dim=dataset.input_dim, num_classes=dataset.num_classes)
    base_model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    base_model.to(device)

    deep_ensemble_models = []
    for i in range(5): 
        model = load_model(model_name, input_dim=dataset.input_dim, num_classes=dataset.num_classes)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        deep_ensemble_models.append(model)

    if not deep_ensemble_models:
        print("dkip Deep Ensemble due to initialization error.", flush=True)

    methods = {
        "MC Dropout": MonteCarloDropout(base_model).to(device),
        "Deep Ensemble": DeepEnsemble(models=deep_ensemble_models).to(device) if deep_ensemble_models else None,
        "Temperature Scaling": TemperatureScaling().to(device) 
    }

    results = {}
    for method_name, model in methods.items():
        if model is None:
            print(f"skipping {method_name} due to initialization error.", flush=True)
            continue

        print(f"eval {method_name}...", flush=True)
        model.eval()

        all_probs, all_logits, all_labels = [], [], []
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                output = model(inputs) if method_name != "Deep Ensemble" else model.predict(inputs)
                logits = output[0] if isinstance(output, tuple) else output  

                probs = F.softmax(logits, dim=-1)

                all_logits.append(logits.cpu())
                all_probs.append(probs.cpu())
                all_labels.append(labels.cpu())

        all_probs = torch.cat(all_probs)
        all_logits = torch.cat(all_logits)
        all_labels = torch.cat(all_labels)

        num_classes = all_probs.shape[1]
        if num_classes != dataset.num_classes:
            print(f"{dataset.num_classes} classes,  got {num_classes}. Adjusting num_classes.", flush=True)

        ece = compute_ece(all_probs, all_labels)
        nll = compute_nll(all_logits, all_labels)
        brier = compute_brier_score(all_probs, all_labels, num_classes=num_classes)

        results[method_name] = {"ECE": ece, "NLL": nll, "Brier Score": brier}

    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/compare_{model_name}_{dataset_name}.txt"

    with open(log_file, "w") as f:
        f.write(f"compare uncertainty for {model_name} on {dataset_name}:\n")
        for method, metrics in results.items():
            f.write(f"{method}:\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value:.4f}\n")
            f.write("\n")

    print(f"\nComparison results saved to {log_file}")






if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare Uncertainty Estimation Methods")
    parser.add_argument("--model", type=str, required=True, choices=["resnet", "densenet"])
    parser.add_argument("--dataset", type=str, required=True, choices=["medical", "financial", "security"])
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    compare_uncertainty_methods(args.model, args.dataset, args.checkpoint)
