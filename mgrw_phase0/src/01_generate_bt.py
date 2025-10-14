# src/01_generate_bt.py
"""
Back-translate English->French->English using MarianMT models (Helsinki).
Caches outputs to data/bt/en_fr_en.txt
"""
import torch
import os
from transformers import MarianTokenizer, MarianMTModel
from tqdm import tqdm

SRC = "data/raw/en_sentences.txt"
OUTDIR = "data/bt"
os.makedirs(OUTDIR, exist_ok=True)
OUTF = os.path.join(OUTDIR, "en_fr_en.txt")

# model names
EN_FR = "Helsinki-NLP/opus-mt-en-fr"
FR_EN = "Helsinki-NLP/opus-mt-fr-en"

def batch_translate(texts, model_name):
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    print(f"Using device: {device}")
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    res = []
    batch = []
    for t in tqdm(texts):
        batch.append(t)
        if len(batch) >= 16:
            encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
            out = model.generate(**encoded, max_length=256)
            res.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
            batch = []
    if batch:
        encoded = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        out = model.generate(**encoded, max_length=256)
        res.extend(tokenizer.batch_decode(out, skip_special_tokens=True))
    return res

def main():
    with open(SRC, encoding="utf-8") as f:
        sents = [l.strip() for l in f if l.strip()]
    print("Loaded", len(sents), "sentences")
    # translate en->fr
    print("Translating EN->FR ...")
    fr_texts = batch_translate(sents, EN_FR)
    print("Translating FR->EN ...")
    en_back = batch_translate(fr_texts, FR_EN)
    with open(OUTF, "w", encoding="utf-8") as f:
        for t in en_back:
            f.write(t.replace("\n"," ") + "\n")
    print("Wrote back-translated to", OUTF)

if __name__ == "__main__":
    main()