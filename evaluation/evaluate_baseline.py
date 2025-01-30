import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.model_loader import load_model
from data.medical_loader import StrokeDataset
from data.financial_loader import FinancialDistressDataset
from data.security_loader import MalwareDataset
from utils.metrics import compute_metrics
import os

def evaluate_baseline(model_name, dataset_name, checkpoint_path):
    print(f"eval: {model_name} on {dataset_name}...", flush=True)

    dataset = StrokeDataset() if dataset_name == "medical" else (
        FinancialDistressDataset() if dataset_name == "financial" else MalwareDataset()
    )

    test_loader = DataLoader(dataset.test_data, batch_size=32, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = load_model(model_name, input_dim=dataset.input_dim, num_classes=dataset.num_classes)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    metrics = compute_metrics(all_preds, all_labels)

    os.makedirs("logs", exist_ok=True)
    log_file = f"logs/baseline_{model_name}_{dataset_name}.txt"

    with open(log_file, "w") as f:
        f.write(f"Evaluation Metrics for {model_name} on {dataset_name}:\n")
        for key, value in metrics.items():
            f.write(f"{key.capitalize()}: {value:.4f}\n")

    print(f"\nEvaluation Metrics saved to {log_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Baseline Model")
    parser.add_argument("--model", type=str, required=True, choices=["resnet", "densenet"])
    parser.add_argument("--dataset", type=str, required=True, choices=["medical", "financial", "security"])
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    evaluate_baseline(args.model, args.dataset, args.checkpoint)
