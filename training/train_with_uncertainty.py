import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import sys
from models.model_loader import load_model
from data.medical_loader import StrokeDataset
from data.financial_loader import FinancialDistressDataset
from data.security_loader import MalwareDataset
from uncertainty.monte_carlo_dropout import MonteCarloDropout
from uncertainty.deep_ensembles import DeepEnsemble
from uncertainty.temperature_scaling import TemperatureScaling
from uncertainty.bayesians_nn import BayesianLinear
from torch.utils.tensorboard import SummaryWriter

sys.stdout.reconfigure(line_buffering=True)

def get_uncertainty_model(base_model, method, dataset, device):
    print(f"init {method} model...", flush=True)
    
    if method == "mc_dropout":
        return MonteCarloDropout(base_model).to(device)
    
    elif method == "deep_ensemble":
        ensemble = DeepEnsemble(args.model, dataset.input_dim, dataset.num_classes)
        for model in ensemble.models:
            model.to(device)
        return ensemble
    
    elif method == "bayesian_nn":
        return BayesianLinear(dataset.input_dim, dataset.num_classes).to(device)
    
    elif method == "temperature_scaling":
        return TemperatureScaling().to(device)
    
    else:
        raise ValueError(f"dont know uncertainty estimation method: {method}")

parser = argparse.ArgumentParser(description="Train Uncertainty-Aware Models")
parser.add_argument("--dataset", type=str, choices=["medical", "financial", "security"], required=True)
parser.add_argument("--model", type=str, choices=["resnet", "densenet"], required=True)
parser.add_argument("--method", type=str, choices=["mc_dropout", "deep_ensemble", "bayesian_nn", "temperature_scaling"], required=True)
parser.add_argument("--epochs", type=int, default=10)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--learning_rate", type=float, default=0.001)
parser.add_argument("--save_path", type=str, default="checkpoints")

args = parser.parse_args()

print(f"load dataset: {args.dataset}...", flush=True)
dataset = StrokeDataset() if args.dataset == "medical" else (
    FinancialDistressDataset() if args.dataset == "financial" else MalwareDataset()
)

train_loader = DataLoader(dataset.train_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(dataset.test_data, batch_size=args.batch_size, shuffle=False)

print("Dataset loaded", flush=True)

print(f"base model: {args.model}...", flush=True)
base_model = load_model(args.model, input_dim=dataset.input_dim, num_classes=dataset.num_classes)

device = "cuda" if torch.cuda.is_available() else "cpu"
model = get_uncertainty_model(base_model, args.method, dataset, device)

print(f"Model init on {device}. Training...", flush=True)

criterion = nn.CrossEntropyLoss()

if args.method == "deep_ensemble":
    optimizers = [optim.Adam(m.parameters(), lr=args.learning_rate) for m in model.models]
    schedulers = [torch.optim.lr_scheduler.StepLR(opt, step_size=10, gamma=0.5) for opt in optimizers]
else:
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

writer = SummaryWriter()

for epoch in range(args.epochs):
    if args.method == "deep_ensemble":
        model.train(train_loader, args.epochs)  

    total_loss, correct, total = 0, 0, 0

    print(f"\nEpoch {epoch + 1}/{args.epochs}...", flush=True)

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)

        if args.method == "deep_ensemble":
            batch_losses = []
            for i, (m, opt) in enumerate(zip(model.models, optimizers)):
                opt.zero_grad()
                outputs = m(inputs)  
                loss = criterion(outputs, labels)
                loss.backward()
                opt.step()
                batch_losses.append(loss.item())
            
            batch_loss = sum(batch_losses) / len(batch_losses)

        else:
            optimizer.zero_grad()

            if args.method == "bayesian_nn" or args.method == "temperature_scaling":
                outputs = model(inputs)
                uncertainty = None
            else:
                outputs, uncertainty = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()

        total_loss += batch_loss
        batch_correct = (outputs.argmax(dim=1) == labels).sum().item()
        correct += batch_correct
        total += labels.size(0)

        print(f"Batch {batch_idx + 1}/{len(train_loader)} - Loss: {batch_loss:.4f} - Accuracy: {batch_correct}/{labels.size(0)}", flush=True)

    epoch_loss = total_loss / total
    epoch_accuracy = correct / total

    if args.method == "deep_ensemble":
        for scheduler in schedulers:
            scheduler.step()
    else:
        scheduler.step()
    
    writer.add_scalar("Loss/train", epoch_loss, epoch)
    writer.add_scalar("Accuracy/train", epoch_accuracy, epoch)

    print(f"Epoch {epoch + 1} completed. Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}", flush=True)

checkpoint_path = f"{args.save_path}/{args.model}_{args.dataset}_{args.method}.pth"

if args.method == "deep_ensemble":
    for i, m in enumerate(model.models):
        torch.save(m.state_dict(), f"{checkpoint_path}_model_{i}.pth")
else:
    torch.save(base_model.state_dict(), checkpoint_path)

writer.close()

print(f"\nModel saved to {checkpoint_path}", flush=True)
