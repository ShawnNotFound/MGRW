# src/gen_watermarked.py
import os, json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
os.makedirs("outputs/watermarked", exist_ok=True)
MODEL_DIR = "outputs/injector_robust"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).to("cuda" if torch.cuda.is_available() else "cpu")

def load_jsonl(p):
    out=[]
    with open(p, encoding="utf-8") as f:
        for l in f:
            out.append(json.loads(l))
    return out

for split in ["val","test","train"]:
    items = load_jsonl(f"data/pairs/{split}.jsonl")
    outp = []
    for it in items:
        inp = f"watermark: {it['bits']} ||| {it['src']}"
        enc = tokenizer(inp, return_tensors="pt", truncation=True, padding=True).to(model.device)
        gen = model.generate(**enc, max_new_tokens=192, num_beams=2)
        txt = tokenizer.decode(gen[0], skip_special_tokens=True)
        outp.append({"text": txt, "bits": it['bits'], "variant": it.get("variant","orig")})
    with open(f"outputs/watermarked/{split}_wm.jsonl","w",encoding="utf-8") as fo:
        for r in outp:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("Wrote outputs/watermarked/" + split + "_wm.jsonl")
