import numpy as np

def make_label_skew_splits(y, num_clients=10, skew=0.7, seed=42):
    rng = np.random.RandomState(seed)
    idx = np.arange(len(y))
    splits = [[] for _ in range(num_clients)]
    for cls in np.unique(y):
        cls_idx = idx[y==cls]
        rng.shuffle(cls_idx)
        main = int(skew*len(cls_idx))
        main_client = rng.randint(num_clients)
        splits[main_client].extend(cls_idx[:main].tolist())
        rest = cls_idx[main:]
        for i, idv in enumerate(rest):
            splits[(main_client+1+i)%num_clients].append(idv)
    return [np.array(s) for s in splits]

def save_splits(splits, path):
    import json
    js = [s.tolist() for s in splits]
    with open(path, "w") as f: f.write(json.dumps(js)) 