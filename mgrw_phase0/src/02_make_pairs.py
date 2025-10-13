# src/02_make_pairs.py
"""
Construct (source, bits) training JSONL for injector training.
Also prepares detector training pairs (watermarked text will be created after injector training).
Outputs:
- data/pairs/train.jsonl
- data/pairs/val.jsonl
- data/pairs/test.jsonl
We use simple random bits per sentence (pilot: 8 bits)
"""
import os, json, random
os.makedirs("data/pairs", exist_ok=True)

SRC = "data/raw/en_sentences.txt"
BT = "data/bt/en_fr_en.txt"  # optional: used to include BT variants in training pairs if desired

def read_lines(p):
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def random_bits(k=8):
    return "".join(random.choice("01") for _ in range(k))

def main():
    sents = read_lines(SRC)
    bt = read_lines(BT)
    N = len(sents)
    K = 8  # bits per sentence
    pairs = []
    for i, s in enumerate(sents):
        bits = random_bits(K)
        pairs.append({"src": s, "bits": bits})
        # optionally add BT variant as training sample with same bits to help robustness
        if i < len(bt):
            pairs.append({"src": bt[i], "bits": bits})
    random.shuffle(pairs)
    n = len(pairs)
    ntrain = int(n*0.8)
    nval = int(n*0.1)
    train = pairs[:ntrain]
    val = pairs[ntrain:ntrain+nval]
    test = pairs[ntrain+nval:]
    for name, arr in [("train","data/pairs/train.jsonl"), ("val","data/pairs/val.jsonl"), ("test","data/pairs/test.jsonl")]:
        with open(f"data/pairs/{name}.jsonl","w",encoding="utf-8") as f:
            subset = {"train":train,"val":val,"test":test}[name]
            for item in subset:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print("Wrote", name, "size:", len({"train":train,"val":val,"test":test}[name]))

if __name__ == "__main__":
    main()