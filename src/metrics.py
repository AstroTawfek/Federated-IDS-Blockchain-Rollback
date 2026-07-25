import os
class RoundLogger:
    def __init__(self, path="experiments/metrics.csv"):
        self.path=path; open(self.path,'w').write("round,acc,excluded,anchor_bytes\n")
    def log(self, r, acc, excluded):
        ab = os.path.getsize("experiments/anchors.jsonl") if os.path.exists("experiments/anchors.jsonl") else 0
        with open(self.path,'a') as f: f.write(f"{r},{acc},{excluded},{ab}\n") 