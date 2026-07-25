import torch, torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, hidden=128, out_dim=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden//2), nn.ReLU(),
            nn.Linear(hidden//2, out_dim)
        )
    def forward(self, x): return self.net(x) 