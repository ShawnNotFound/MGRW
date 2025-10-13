# src/04_generate_watermarked.py
"""
Load injector checkpoint, generate watermarked sentences for val/test,
and save detector training jsonl: data/pairs/train_detector.jsonl, val_detector.jsonl, test_detector.jsonl
Each record: {"text": "<watermarked>", "bits":"0101..."}
Also generate BT variants of watermarked text (use earlier BT model or re-run Marian)
"""
import os, json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm import tqdm

INJECTOR_DIR = "outputs/injector"
PAIRS_DIR = "data/pairs"
OUT_WM = "outputs/watermarked"
os.makedirs(OUT_WM, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(INJECTOR_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(INJECTOR_DIR).to("cuda")

def load_jsonl(p):
    arr=[]
    with open(p, encoding="utf-8") as f:
        for l in f:
            arr.append(json.loads(l))
    return arr

def generate_one(src, bits):
    inp = f"watermark: {bits} ||| {src}"
    enc = tokenizer(inp, return_tensors="pt", truncation=True, padding=True, max_length=192).to("cuda")
    out = model.generate(**enc, max_new_tokens=192, num_beams=4)
    txt = tokenizer.decode(out[0], skip_special_tokens=True)
    return txt

def main():
    for split in ["val","test","train"]:
        items = load_jsonl(os.path.join(PAIRS_DIR,f"{split}.jsonl"))
        outf = os.path.join(OUT_WM, f"{split}_wm.jsonl")
        with open(outf, "w", encoding="utf-8") as f:
            for it in tqdm(items, desc=f"gen {split}"):
                wm = generate_one(it["src"], it["bits"])
                record = {"text": wm, "bits": it["bits"]}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        print("Wrote", outf)

    # Optionally: create back-translated variants of watermarked text using data/bt pipeline or Marian
    print("Generation complete. Now create BT variants separately if needed.")

if __name__ == "__main__":
    main()