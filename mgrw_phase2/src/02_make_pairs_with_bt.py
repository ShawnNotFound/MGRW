# src/make_pairs_with_bt.py
import os, json, random
os.makedirs("data/pairs", exist_ok=True)

SRC = "data/raw/en_sentences.txt"
BT_FR = "data/bt/en_fr_en.txt"
BT_DE = "data/bt/en_de_en.txt"

def read_lines(p):
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip()]

def rand_bits(k):
    import random
    return "".join(random.choice("01") for _ in range(k))

def main(bits_per=8):
    src = read_lines(SRC)
    fr = read_lines(BT_FR)
    de = read_lines(BT_DE)
    pairs=[]
    for i, s in enumerate(src):
        bits = rand_bits(bits_per)
        pairs.append({"src": s, "bits": bits, "variant": "orig"})
        if i < len(fr):
            pairs.append({"src": fr[i], "bits": bits, "variant": "frbt"})
        if i < len(de):
            pairs.append({"src": de[i], "bits": bits, "variant": "debt"})
    random.shuffle(pairs)
    n = len(pairs)
    ntrain = int(n*0.8); nval = int(n*0.1)
    splits = {"train": pairs[:ntrain], "val": pairs[ntrain:ntrain+nval], "test": pairs[ntrain+nval:]}
    for k,v in splits.items():
        with open(f"data/pairs/{k}.jsonl","w",encoding="utf-8") as fo:
            for it in v:
                fo.write(json.dumps(it, ensure_ascii=False) + "\n")
        print("Wrote", k, len(v))

if __name__ == "__main__":
    main(bits_per=8)
