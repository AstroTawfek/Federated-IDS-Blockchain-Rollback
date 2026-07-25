import numpy as np, torch, torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from copy import deepcopy
from src.model import MLP
import sys, os
from src.data import load_dataset 

def client_train(model, Xc, yc, epochs=2, batch=256, lr=1e-3, device='cpu'):
    m = deepcopy(model).to(device)
    opt = torch.optim.Adam(m.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    loader = DataLoader(TensorDataset(torch.tensor(Xc), torch.tensor(yc)), batch_size=batch, shuffle=True)
    m.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(m(xb), yb); loss.backward(); opt.step()
    return m.state_dict()

def fedavg(states):
    avg = deepcopy(states[0])
    for k in avg.keys():
        for sd in states[1:]:
            avg[k] += sd[k]
        avg[k] /= len(states)
    return avg

def evaluate(model, X, y, batch=512, device='cpu'):
    model.to(device); model.eval()
    loader = DataLoader(TensorDataset(torch.tensor(X), torch.tensor(y)), batch_size=batch)
    correct=tot=0
    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb).argmax(1)
            correct += (pred==yb).sum().item(); tot += len(yb)
    return correct/tot

import numpy as np

import numpy as np

def make_client_splits(X, y, num_clients=5):
    """
    Split dataset into num_clients roughly equal parts.
    Each part is returned as (Xc, yc).
    """
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    X, y = X[idx], y[idx]

    splits = []
    size = len(y) // num_clients
    for i in range(num_clients):
        start = i * size
        end = (i+1) * size if i < num_clients-1 else len(y)
        splits.append((X[start:end], y[start:end]))
    return splits


if __name__ == "__main__":
    # --- Argument parsing ---
    if len(sys.argv) < 2:
        raise ValueError("Please specify dataset: nsl or cic")

    ds = sys.argv[1].lower()
    if ds not in ["nsl", "cic"]:
        raise ValueError("Unknown dataset argument. Use 'nsl' or 'cic'.")

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset arrays
    X_train, y_train, X_test, y_test = load_dataset(ds)

    # Create client splits (e.g. 5 clients)
    client_splits = make_client_splits(X_train, y_train, num_clients=5)

    # Initialize global model
    global_model = MLP(X_train.shape[1], hidden=128).to(device)

    # Federated loop
    for round in range(5):
        client_states = []
        for (Xc, yc) in client_splits:
            sd = client_train(global_model, Xc, yc, device=device)
            client_states.append(sd)

        avg_state = fedavg(client_states)
        global_model.load_state_dict(avg_state)

        acc = evaluate(global_model, X_test, y_test, device=device)
        print(f"[{ds.upper()}] Round {round+1} accuracy: {acc:.4f}")

    # --- Save final federated model ---
    os.makedirs("BFIPPDSR_project/experiments", exist_ok=True)
    torch.save(global_model.state_dict(),
               f"BFIPPDSR_project/experiments/final_model_{ds}.pth")
    print(f"Final federated model saved to experiments/final_model_{ds}.pth")