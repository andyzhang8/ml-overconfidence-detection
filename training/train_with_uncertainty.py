import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from models.model_loader import load_model
from data.medical_loader import StrokeDataset
from data.financial_loader import FinancialDistressDataset
from data.security_loader import MalwareDataset
from uncertainty.monte_carlo_dropout import MonteCarloDropout
from uncertainty.deep_ensembles import DeepEnsemble
from uncertainty.temperature_scaling import TemperatureScaling
from uncertainty.bayesians_nn import BayesianLinear
from torch.utils.tensorboard import SummaryWriter

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

parser = argparse.ArgumentParser(description="Train Uncertainty-Aware Models")
parser.add_argument("--dataset", type=str, choices=["medical", "financial", "security"], required=True)
parser.add_argument("--model", type=str, choices=["resnet", "densenet"], required=True)
parser.add_argument("--method", type=str, choices=["mc_dropout", "deep_ensemble", "bayesian_nn", "temperature_scaling"], required=True)
parser.add_argument("--epochs", type=int, default=50)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--save_path", type=str, default="checkpoints")

args = parser.parse_args()

dataset = StrokeDataset() if args.dataset == "medical" else (
    FinancialDistressDataset() if args.dataset == "financial" else MalwareDataset()
)

train_loader = DataLoader(dataset.train_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(dataset.test_data, batch_size=args.batch_size, shuffle=False)

base_model = load_model(args.model, input_dim=dataset.input_dim, num_classes=dataset.num_classes)
model = get_uncertainty_model(base_model, args.method, dataset)
model = model.to("cuda" if torch.cuda.is_available() else "cpu")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
writer = SummaryWriter()

for epoch in range(args.epochs):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to("cuda"), labels.to("cuda")

        optimizer.zero_grad()
        outputs, uncertainty = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    scheduler.step()
    writer.add_scalar("Loss/train", total_loss / total, epoch)
    writer.add_scalar("Accuracy/train", correct / total, epoch)

torch.save(model.state_dict(), f"{args.save_path}/{args.model}_{args.dataset}_{args.method}.pth")
writer.close()
