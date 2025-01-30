import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import joblib
import os
from datetime import datetime
from models.model_loader import load_model
from uncertainty.monte_carlo_dropout import MonteCarloDropout
from uncertainty.deep_ensembles import DeepEnsemble
from uncertainty.temperature_scaling import TemperatureScaling
from uncertainty.bayesians_nn import BayesianLinear
from data.medical_loader import StrokeDataset
from data.financial_loader import FinancialDistressDataset
from data.security_loader import MalwareDataset

clf = joblib.load("overconfidence_classifier.pkl")

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

def predict_overconfidence(model_name, dataset_name, method, checkpoint_path):
    print(f"Predicting Overconfidence for {model_name} on {dataset_name}...", flush=True)

    # 1. Load dataset
    if dataset_name == "medical":
        dataset = StrokeDataset()
    elif dataset_name == "financial":
        dataset = FinancialDistressDataset()
    elif dataset_name == "security":
        dataset = MalwareDataset()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    test_loader = DataLoader(dataset.test_data, batch_size=32, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    base_model = load_model(model_name, input_dim=dataset.input_dim, num_classes=dataset.num_classes)
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        print("Extracting model state_dict from 'model' key...", flush=True)
        checkpoint = checkpoint["model"]

    base_model.load_state_dict(checkpoint, strict=False)
    base_model.to(device)
    base_model.eval()

    model = get_uncertainty_model(base_model, method, dataset, checkpoint_path)
    model.to(device)
    model.eval()

    print("Running inference...", flush=True)

    os.makedirs("logs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = f"logs/ai_overconfidence_{model_name}_{dataset_name}_{method}_{timestamp}.csv"

    with open(output_csv, "w") as f:
        f.write("SampleID,Pred,Confidence,Uncertainty,TrueLabel,OverconfProb,OverconfLabel\n")

    sample_global_idx = 0
    for batch_idx, (inputs, labels) in enumerate(test_loader):
        print(f"Processing batch {batch_idx + 1}/{len(test_loader)}...", flush=True)
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            if method == "deep_ensemble":
                logits, uncertainty = model.predict(inputs)
            elif method in ["mc_dropout", "bayesian_nn"]:
                logits, uncertainty = model(inputs)
            else:
                logits = model(inputs)
                uncertainty = torch.zeros_like(logits)

        probs = F.softmax(logits, dim=-1)
        max_probs, preds = torch.max(probs, dim=1)

        if uncertainty.ndim > 1:
            uncertainty = uncertainty.mean(dim=1)

        batch_rows = []
        for i in range(len(labels)):
            sample_id = sample_global_idx + i
            conf_val = max_probs[i].item()
            unc_val = uncertainty[i].item()
            true_label = labels[i].item()
            pred_label = preds[i].item()

            overconf_prob = clf.predict_proba([[conf_val, unc_val, pred_label, dataset_name, model_name, method]])[0][1]
            overconf_label = int(overconf_prob >= 0.5)

            batch_rows.append({
                "SampleID": sample_id,
                "Pred": pred_label,
                "Confidence": conf_val,
                "Uncertainty": unc_val,
                "TrueLabel": true_label,
                "OverconfProb": overconf_prob,
                "OverconfLabel": overconf_label
            })

        sample_global_idx += len(labels)

        with open(output_csv, "a") as f:
            for row in batch_rows:
                f.write("{},{},{:.4f},{:.4f},{},{:.4f},{}\n".format(
                    row["SampleID"],
                    row["Pred"],
                    row["Confidence"],
                    row["Uncertainty"],
                    row["TrueLabel"],
                    row["OverconfProb"],
                    row["OverconfLabel"]
                ))

    print(f"\nAI Overconfidence results saved to '{output_csv}'", flush=True)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Predict AI Overconfidence")
    parser.add_argument("--model", type=str, required=True, choices=["resnet", "densenet"])
    parser.add_argument("--dataset", type=str, required=True, choices=["medical", "financial", "security"])
    parser.add_argument("--method", type=str, required=True, choices=["mc_dropout", "deep_ensemble", "bayesian_nn", "temperature_scaling"])
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    predict_overconfidence(args.model, args.dataset, args.method, args.checkpoint)
