import sys
import torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from src.model import MLP
from src.data import load_dataset

def train_eval(dataset="nsl", epochs=8, batch=512, lr=1e-3, hidden=128):
    # Load dataset arrays
    Xtr, ytr, Xte, yte = load_dataset(dataset)

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model, optimizer, loss
    model = MLP(Xtr.shape[1], hidden=hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()

    # Data loaders
    train_loader = DataLoader(
        TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
        batch_size=batch, shuffle=True
    )
    test_loader = DataLoader(
        TensorDataset(torch.tensor(Xte), torch.tensor(yte)),
        batch_size=batch
    )

    # Training loop
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()

    # Evaluation
    model.eval()
    correct, tot = 0, 0
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += (pred == yb).sum().item()
            tot += len(yb)

    print(f"[{dataset.upper()}] Centralized test accuracy: {correct/tot:.4f}")
    return model

if __name__ == "__main__":
    # --- Argument parsing ---
    if len(sys.argv) < 2:
        raise ValueError("Please specify dataset: nsl or cic")

    ds = sys.argv[1].lower()

    if ds not in ["nsl", "cic"]:
        raise ValueError("Unknown dataset argument. Use 'nsl' or 'cic'.")

    # Run training/evaluation
    model = train_eval(dataset=ds)

    # --- Save trained model ---
    import os
    os.makedirs("BFIPPDSR_project/experiments", exist_ok=True)
    torch.save(
        model.state_dict(),
        f"BFIPPDSR_project/experiments/final_model_{ds}.pth"
    )
    print(f"Saved model to experiments/final_model_{ds}.pth") 