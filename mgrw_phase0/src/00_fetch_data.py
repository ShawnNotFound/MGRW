# src/00_fetch_data.py
"""
Fetch/generate a small English sentence dataset (pilot). Strategy:
- Use datasets 'wikipedia' or 'cc_news' if available.
- Fallback: use a small synthetic set if download fails.
Output: data/raw/en_sentences.txt (one sentence per line), N=5000
"""
import os
from datasets import load_dataset
from tqdm import tqdm

OUT = "data/raw"
os.makedirs(OUT, exist_ok=True)
OUTF = os.path.join(OUT, "en_sentences.txt")
N = 5000

def main():
    sents = []
    try:
        ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
        for item in tqdm(ds.select(range(0, 300000)).iter(batch_size=1000), desc="scan wikitext"):
            for line in item["text"]:
                line = line.strip()
                if 20 < len(line) < 200 and "." in line:
                    # 过滤非自然语言的无效行，如编号或军队单位名称
                    if any(bad in line for bad in ["No.", "Squadron", "Regiment", "Brigade", "Flight", "Battalion"]):
                        continue
                    if not line[0].isalpha():
                        continue
                    if sum(c.isdigit() for c in line) > 5:
                        continue
                    sents.append(line)
                    if len(sents) >= N:
                        break
            if len(sents) >= N:
                break
    except Exception as e:
        print("wikipedia load failed:", e)
    if len(sents) < N:
        print("Falling back to cc_news or synthetic")
        try:
            ds = load_dataset("cc_news", split="train")
            for item in tqdm(ds.select(range(0, 200000)).iter(batch_size=1000), desc="scan cc_news"):
                text = item["text"]
                if text and 20 < len(text) < 200:
                    sents.append(text.strip())
                    if len(sents) >= N:
                        break
        except Exception as e:
            print("cc_news failed:", e)
    if len(sents) < N:
        raise RuntimeError(f"Only {len(sents)} valid sentences found — not enough natural data to reach {N}.")
    # truncate & write
    sents = sents[:N]
    with open(OUTF, "w", encoding="utf-8") as f:
        for s in sents:
            f.write(s.replace("\n"," ") + "\n")
    print("Wrote", len(sents), "sentences to", OUTF)

if __name__ == "__main__":
    main()