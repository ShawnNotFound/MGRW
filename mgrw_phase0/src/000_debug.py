import json
from collections import Counter

with open("../data/train.json", "r") as f:
    data = [json.loads(line) for line in f]

print(Counter([d["label"] for d in data]))
