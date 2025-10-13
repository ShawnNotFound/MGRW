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
        # try wikipedia (en) short snippets
        ds = load_dataset("wikipedia", "20220301.en", split="train")
        # extract short sentences from article_text
        for item in tqdm(ds.select(range(0, 20000)).iter(batch_size=1000), desc="scan wiki"):
            for text in item["text"]:
                for line in text.split("\n"):
                    line = line.strip()
                    if 20 < len(line) < 200 and "." in line:
                        sents.append(line)
                        if len(sents) >= N:
                            break
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
            for item in tqdm(ds.select(range(0, 20000)).iter(batch_size=1000), desc="scan cc_news"):
                text = item["text"]
                if text and 20 < len(text) < 200:
                    sents.append(text.strip())
                    if len(sents) >= N:
                        break
        except Exception as e:
            print("cc_news failed:", e)
    if len(sents) < N:
        # synthetic fallback
        print("Using synthetic fallback (repeating short sentence)")
        base = "This is a sample sentence used for pilot experiments."
        while len(sents) < N:
            sents.append(base)
    # truncate & write
    sents = sents[:N]
    with open(OUTF, "w", encoding="utf-8") as f:
        for s in sents:
            f.write(s.replace("\n"," ") + "\n")
    print("Wrote", len(sents), "sentences to", OUTF)

if __name__ == "__main__":
    main()