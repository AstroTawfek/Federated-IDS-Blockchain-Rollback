import numpy as np, torch

class UpdateStats:
    def __init__(self): self.means={}
    def score(self, cid, sd):
        vec = torch.cat([p.flatten() for p in sd.values()]).detach().cpu().numpy()
        m = self.means.get(cid, vec)
        cos = np.dot(vec,m)/(np.linalg.norm(vec)*np.linalg.norm(m)+1e-8)
        norm_ratio = np.linalg.norm(vec)/(np.linalg.norm(m)+1e-8)
        return (1-cos) + abs(np.log(norm_ratio+1e-8))
    def update(self, cid, sd):
        vec = torch.cat([p.flatten() for p in sd.values()]).detach().cpu().numpy()
        m = self.means.get(cid, None)
        self.means[cid] = vec if m is None else 0.9*m + 0.1*vec

def flag(threat, thresh=1.5): return threat>thresh 