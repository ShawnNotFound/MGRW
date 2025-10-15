# src/01_generate_bt_wm.py
"""
Generate back-translated variants for watermarked outputs using MarianMT.
Input: outputs/watermarked/<split>_wm.jsonl
Output: outputs/watermarked/<split>_wm_bt.jsonl
"""
import os, json
from transformers import MarianTokenizer, MarianMTModel
from tqdm import tqdm

EN_FR = "Helsinki-NLP/opus-mt-en-fr"
FR_EN = "Helsinki-NLP/opus-mt-fr-en"
DEVICE = "cuda" if __import__("torch").cuda.is_available() else "cpu"

def batch_translate(texts, model_name, device=DEVICE, batch_size=8):
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name).to(device)
    outs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        gen = model.generate(**enc, max_length=256)
        outs.extend(tokenizer.batch_decode(gen, skip_special_tokens=True))
    return outs

def process_split(split):
    inpath = f"outputs/watermarked/{split}_wm.jsonl"
    if not os.path.exists(inpath):
        print("Missing", inpath)
        return
    outpath = f"outputs/watermarked/{split}_wm_bt.jsonl"
    texts=[]
    records=[]
    with open(inpath, encoding="utf-8") as f:
        for l in f:
            rec=json.loads(l)
            texts.append(rec["text"])
            records.append(rec)
    print(f"Loaded {len(texts)} watermarked items for split {split}")
    print("Translating EN->FR ...")
    fr = batch_translate(texts, EN_FR)
    print("Translating FR->EN ...")
    en_back = batch_translate(fr, FR_EN)
    with open(outpath, "w", encoding="utf-8") as f:
        for rec, back in zip(records, en_back):
            newrec = {"text": back, "bits": rec["bits"]}
            f.write(json.dumps(newrec, ensure_ascii=False) + "\n")
    print("Wrote BT file:", outpath)

if __name__ == "__main__":
    for sp in ["train", "val", "test"]:
        process_split(sp)
