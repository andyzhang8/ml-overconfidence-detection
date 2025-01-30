import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from models.model_loader import load_model
from data.medical_loader import StrokeDataset
from data.financial_loader import FinancialDistressDataset
from data.security_loader import MalwareDataset
from utils.metrics import compute_metrics

def evaluate_baseline(model_name, dataset_name, checkpoint_path):
    dataset = StrokeDataset() if dataset_name == "medical" else (
        FinancialDistressDataset() if dataset_name == "financial" else MalwareDataset()
    )

    test_loader = DataLoader(dataset.test_data, batch_size=32, shuffle=False)

    model = load_model(model_name, input_dim=dataset.input_dim, num_classes=dataset.num_classes)
    model.load_state_dict(torch.load(checkpoint_path))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to("cuda"), labels.to("cuda")
            outputs = model(inputs)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    compute_metrics(all_preds, all_labels)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate Baseline Model")
    parser.add_argument("--model", type=str, required=True, choices=["resnet", "densenet"])
    parser.add_argument("--dataset", type=str, required=True, choices=["medical", "financial", "security"])
    parser.add_argument("--checkpoint", type=str, required=True)
    args = parser.parse_args()

    evaluate_baseline(args.model, args.dataset, args.checkpoint)
