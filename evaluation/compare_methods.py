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

def compare_uncertainty_methods(model_name, dataset_name, checkpoint_path):
    dataset = StrokeDataset() if dataset_name == "medical" else (
        FinancialDistressDataset() if dataset_name == "financial" else MalwareDataset()
    )

    test_loader = DataLoader(dataset.test_data, batch_size=32, shuffle=False)

    base_model = load_model(model_name, input_dim=dataset.input_dim, num_classes=dataset.num_classes)
    base_model.load_state_dict(torch.load(checkpoint_path))

    mc_dropout_model = MonteCarloDropout(base_model)
    deep_ensemble_model = DeepEnsemble(model_name, dataset.input_dim, dataset.num_classes, num_models=5)
    temp_scaling_model = TemperatureScaling()
    
    methods = {
        "MC Dropout": mc_dropout_model,
        "Deep Ensemble": deep_ensemble_model,
        "Temperature Scaling": temp_scaling_model
    }

    results = {}
    for method_name, model in methods.items():
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

        results[method_name] = {"ECE": ece, "NLL": nll, "Brier Score": brier}

    print("Comparison of Uncertainty Methods:")
    for method, metrics in results.items():
        print(f"{method}: ECE: {metrics['ECE']:.4f}, NLL: {metrics['NLL']:.4f}, Brier Score: {metrics['Brier Score']:.4f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Compare Uncertainty Estimation Methods")
    parser.add_argument("--model", type=str, required=True, choices=["resnet", "densenet"])
    parser.add_argument("--dataset", type=str, required=True, choices=["medical", "financial", "security"])
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    compare_uncertainty_methods(args.model, args.dataset, args.checkpoint)
