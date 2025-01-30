import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from models.model_loader import load_model
from data.medical_loader import StrokeDataset
from data.financial_loader import FinancialDistressDataset
from data.security_loader import MalwareDataset
from utils.metrics import compute_metrics
from torch.utils.tensorboard import SummaryWriter

parser = argparse.ArgumentParser(description="Train Baseline Models")
parser.add_argument("--dataset", type=str, choices=["medical", "financial", "security"], required=True, help="Dataset to use")
parser.add_argument("--model", type=str, choices=["resnet", "densenet", "transformer"], required=True, help="Model to train")
parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
parser.add_argument("--learning_rate", type=float, default=0.001, help="Learning rate")
parser.add_argument("--save_path", type=str, default="checkpoints", help="Path to save model")
args = parser.parse_args()

if args.dataset == "medical":
    dataset = StrokeDataset()
elif args.dataset == "financial":
    dataset = FinancialDistressDataset()
elif args.dataset == "security":
    dataset = MalwareDataset()

train_loader = DataLoader(dataset.train_data, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(dataset.test_data, batch_size=args.batch_size, shuffle=False)

model = load_model(args.model, input_dim=dataset.input_dim, num_classes=dataset.num_classes)
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
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct += (outputs.argmax(dim=1) == labels).sum().item()
        total += labels.size(0)

    scheduler.step()
    writer.add_scalar("Loss/train", total_loss / total, epoch)
    writer.add_scalar("Accuracy/train", correct / total, epoch)

    print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/total:.4f}, Accuracy: {correct/total:.4f}")

model.eval()
compute_metrics(model, test_loader)

torch.save(model.state_dict(), f"{args.save_path}/{args.model}_{args.dataset}.pth")
writer.close()
