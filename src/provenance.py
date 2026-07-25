import hashlib, json, time, torch

def sha256(b): return hashlib.sha256(b).hexdigest()

def merkle_root(state_dict):
    leaves = []
    for k,v in state_dict.items():
        leaves.append(sha256(v.detach().cpu().numpy().tobytes()))
    nodes = leaves[:]
    if not nodes: return sha256(b'')
    while len(nodes)>1:
        nxt=[]
        for i in range(0,len(nodes),2):
            a = nodes[i]; b = nodes[i+1] if i+1<len(nodes) else a
            nxt.append(sha256((a+b).encode()))
        nodes = nxt
    return nodes[0]

def log_anchor(round_id, client_id, state_dict, path="experiments/anchors.jsonl"):
    rec = {"round": round_id, "client": client_id, "root": merkle_root(state_dict), "ts": time.time()}
    with open(path, "a") as f: f.write(json.dumps(rec)+"\n") 