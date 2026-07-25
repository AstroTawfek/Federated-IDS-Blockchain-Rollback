# run_experiment.py
import json, sys
import torch
from src.data import load_dataset
from src.model import MLP
from src.splits import make_label_skew_splits
from src.federated import client_train, fedavg, evaluate
from src.attacks import label_flip, byzantine_update, scale_negative
from src.provenance import log_anchor
from src.detect import UpdateStats, flag
from src.metrics import RoundLogger 

def run(cfg):
    Xtr,ytr,Xte,yte = load_dataset(cfg["dataset"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(Xtr.shape[1], hidden=cfg.get("hidden",128)).to(device)
    splits = make_label_skew_splits(ytr, num_clients=cfg["num_clients"], skew=0.7, seed=42)
    stats = UpdateStats()
    logger = RoundLogger(cfg.get("metrics_path","experiments/metrics.csv"))

    attack_clients = set(cfg.get("attack_clients", []))
    attack_type = cfg.get("attack_type", None)

    for r in range(cfg["rounds"]):
        client_states=[]; weights=[]
        for c, idx in enumerate(splits):
            yc = ytr[idx].copy()
            if c in attack_clients and attack_type=='label_flip':
                yc = label_flip(yc, frac=cfg.get("flip_frac",0.3))
            sd = client_train(model, Xtr[idx], yc,
                              epochs=cfg.get("epochs_per_client",2),
                              batch=cfg.get("batch",256), device=device)
            if c in attack_clients and attack_type=='byzantine':
                sd = byzantine_update(sd, scale=cfg.get("noise_scale",10.0))
            if c in attack_clients and attack_type=='negscale':
                sd = scale_negative(sd, scale=cfg.get("neg_scale",5.0))

            log_anchor(r, c, sd)  # provenance logging

            tscore = stats.score(c, sd)
            stats.update(c, sd)
            w = 0.0 if flag(tscore, thresh=cfg.get("detect_thresh",1.5)) else 1.0
            client_states.append(sd); weights.append(w)

        # Weighted FedAvg (rollback by exclusion)
        avg = {}
        for k in client_states[0].keys():
            s = sum(sd[k]*w for sd, w in zip(client_states, weights))
            avg[k] = s / (sum(weights)+1e-8)
        model.load_state_dict(avg)
        acc = evaluate(model, Xte, yte, batch=cfg.get("batch",256), device=device)
        excluded = weights.count(0.0)
        logger.log(r, acc, excluded)
        print(f"[{cfg['dataset'].upper()}] Round {r+1}/{cfg['rounds']} | Acc: {acc:.4f} | Excluded: {excluded}")
    return model

if __name__=="__main__":
    cfg_path = sys.argv[1] if len(sys.argv)>1 else "configs/exp_nsl.json"
    cfg = json.load(open(cfg_path))
    run(cfg)
    print("NSL-KDD Federated Experiment Completed.\n")

    cfg_path = sys.argv[1] if len(sys.argv)>1 else "configs/exp_cic.json"
    cfg = json.load(open(cfg_path))
    run(cfg)
    print("CIC-IDS2017 Federated Experiment Completed.")