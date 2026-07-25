import numpy as np, torch

def label_flip(y, frac=0.3, rng=np.random.RandomState(0)):
    y = y.copy()
    n = int(frac*len(y))
    idx = rng.choice(len(y), n, replace=False)
    y[idx] = 1 - y[idx]
    return y

def byzantine_update(state_dict, scale=10.0):
    sd = {k: v.clone() for k,v in state_dict.items()}
    for k,v in sd.items():
        sd[k] = torch.randn_like(v) * scale
    return sd

def scale_negative(state_dict, scale=5.0):
    return {k: -scale*v for k,v in state_dict.items()} 