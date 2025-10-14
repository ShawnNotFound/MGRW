# src/eval_phase2.py
import json, os, numpy as np
from sklearn.metrics import roc_auc_score
from transformers import AutoTokenizer, AutoModel
import torch

WMDIR = "outputs/watermarked"
DET_MODEL_PATH = "outputs/detector/detector.pt"  # adapt path
ENC = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(ENC)
encoder = AutoModel.from_pretrained(ENC).to("cuda" if torch.cuda.is_available() else "cpu")

def load_jsonl(p):
    arr=[]
    with open(p, encoding="utf-8") as f:
        for l in f:
            arr.append(json.loads(l))
    return arr

# load detector net (your detector implementation)
from src.train_detector import Detector, collate_fn  # if you saved Detector class there

detector = Detector(encoder, nbits=8).to("cuda")
detector.load_state_dict(torch.load(DET_MODEL_PATH))
detector.eval()

def infer_items(items):
    ys=[]; ps=[]
    for it in items:
        toks = tokenizer(it["text"], return_tensors="pt", truncation=True, padding=True, max_length=192).to(detector.encoder.device)
        with torch.no_grad():
            logits = detector(input_ids=toks["input_ids"], attention_mask=toks["attention_mask"]).cpu().numpy()[0]
            probs = 1/(1+np.exp(-logits))
        ys.append([int(ch) for ch in it["bits"]])
        ps.append(probs)
    ys=np.vstack(ys); ps=np.vstack(ps)
    pred = (ps>0.5).astype(int)
    brr = (pred == ys).mean()
    aucs=[]
    for i in range(ys.shape[1]):
        try:
            aucs.append(roc_auc_score(ys[:,i], ps[:,i]))
        except:
            aucs.append(float("nan"))
    return {"brr":float(brr), "auc_mean":float(np.nanmean(aucs))}

# load test watermarked and separate by variant
items = load_jsonl(os.path.join(WMDIR, "test_wm.jsonl"))
orig_items = [x for x in items if x.get("variant","orig")=="orig"]
frbt_items = [x for x in items if x.get("variant")=="frbt"]
debt_items = [x for x in items if x.get("variant")=="debt"]

res_orig = infer_items(orig_items)
res_frbt = infer_items(frbt_items) if frbt_items else None
res_debt = infer_items(debt_items) if debt_items else None

out = {"orig": res_orig, "frbt": res_frbt, "debt": res_debt}
with open("outputs/results_phase2.json","w",encoding="utf-8") as fo:
    json.dump(out, fo, indent=2)
print(json.dumps(out, indent=2))
