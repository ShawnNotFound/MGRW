# src/bt_multi.py
import os
from transformers import MarianTokenizer, MarianMTModel
from tqdm import tqdm

SRC = "data/raw/en_sentences.txt"
OUT_DIR = "data/bt"
os.makedirs(OUT_DIR, exist_ok=True)

PAIRS = [
    ("Helsinki-NLP/opus-mt-en-fr", "Helsinki-NLP/opus-mt-fr-en", "en_fr_en.txt"),
    ("Helsinki-NLP/opus-mt-en-de", "Helsinki-NLP/opus-mt-de-en", "en_de_en.txt"),
]

def batch_translate(texts, model_name, device, batch_size=8):
    tok = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    res=[]
    for i in tqdm(range(0,len(texts),batch_size), desc=f"trans {model_name}"):
        batch = texts[i:i+batch_size]
        enc = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            out = model.generate(**enc, max_length=256)
        res.extend(tok.batch_decode(out, skip_special_tokens=True))
    return res

def main(device="cuda"):
    with open(SRC, encoding="utf-8") as f:
        sents = [l.strip() for l in f if l.strip()][:5000]  # pilot up to 5k
    for en2x, x2en, outname in PAIRS:
        print("Running", en2x, "->", x2en)
        inter = batch_translate(sents, en2x, device)
        back = batch_translate(inter, x2en, device)
        with open(os.path.join(OUT_DIR, outname), "w", encoding="utf-8") as fo:
            for line in back:
                fo.write(line.replace("\n"," ") + "\n")
        print("Wrote", outname)

if __name__ == "__main__":
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    main(dev)
