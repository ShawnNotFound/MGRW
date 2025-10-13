# src/06_eval.py
"""
Compute:
- BRR (bit recovery rate) on test set (original watermarked)
- BRR after back-translation (use data/bt or create bt of watermarked)
- FPR on unwatermarked human text (use data/raw/en_sentences.txt)
Outputs results JSON to outputs/results/phase0_results.json
"""
import os, json, numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score

WM_DIR = "outputs/watermarked"
DET_DIR = "outputs/detector"
RESULTS = "outputs/results"
os.makedirs(RESULTS, exist_ok=True)
MODEL = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
encoder = AutoModel.from_pretrained(MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class DetectorNet(nn.Module):
    def __init__(self, encoder_model, nbits):
        super().__init__()
        self.encoder = encoder_model
        hidden = encoder_model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden, hidden//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden//2, nbits)
        )
    def forward(self, input_ids=None, attention_mask=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        emb = out.last_hidden_state[:,0,:]
        logits = self.classifier(emb)
        return logits

def load_jsonl(p):
    arr=[]
    with open(p, encoding="utf-8") as f:
        for l in f:
            arr.append(json.loads(l))
    return arr

def run_inference(items, model_state_path):
    # items: list of {"text":..., "bits":...}
    nbits = len(items[0]["bits"])
    model = DetectorNet(encoder, nbits).to(device)
    model.load_state_dict(torch.load(model_state_path))
    model.eval()
    all_y, all_p = [], []
    batch_texts=[]
    for it in tqdm(items):
        batch_texts.append(it["text"])
        if len(batch_texts)>=16:
            toks = tokenizer(batch_texts, padding=True, truncation=True, return_tensors="pt", max_length=192).to(device)
            with torch.no_grad():
                logits = model(input_ids=toks["input_ids"], attention_mask=toks["attention_mask"]).cpu().numpy()
                probs = 1/(1+np.exp(-logits))
            for i, bt in enumerate(batch_texts):
                y = [int(ch) for ch in items[0]["bits"]]  # but we must align; simpler to process one-by-one below
            all_p.append(probs)
            batch_texts=[]
    # fallback: process one-by-one to keep simple
    all_y=[]; all_p=[]
    for it in tqdm(items):
        toks = tokenizer(it["text"], return_tensors="pt", truncation=True, padding=True, max_length=192).to(device)
        with torch.no_grad():
            logits = model(input_ids=toks["input_ids"], attention_mask=toks["attention_mask"]).cpu().numpy()[0]
            probs = 1/(1+np.exp(-logits))
        all_p.append(probs)
        all_y.append([int(ch) for ch in it["bits"]])
    all_y = np.vstack(all_y); all_p = np.vstack(all_p)
    pred = (all_p>0.5).astype(int)
    brr = (pred == all_y).mean()
    # per-bit AUC (average)
    aucs=[]
    for i in range(all_y.shape[1]):
        try:
            aucs.append(roc_auc_score(all_y[:,i], all_p[:,i]))
        except:
            aucs.append(float("nan"))
    return {"brr":float(brr), "auc_mean":float(np.nanmean(aucs))}

def main():
    test_items = load_jsonl(os.path.join(WM_DIR,"test_wm.jsonl"))
    model_path = os.path.join(DET_DIR,"detector.pt")
    res_orig = run_inference(test_items, model_path)
    # attempt BT variant: if you made back-translated watermarked file outputs/watermarked/test_wm_bt.jsonl use it
    bt_file = os.path.join(WM_DIR,"test_wm_bt.jsonl")
    res_bt = None
    if os.path.exists(bt_file):
        bt_items = load_jsonl(bt_file)
        res_bt = run_inference(bt_items, model_path)
    # FPR: run detector on unwatermarked human
    human = []
    with open("data/raw/en_sentences.txt", encoding="utf-8") as f:
        for i,l in enumerate(f):
            if i>=1000: break
            human.append({"text":l.strip(), "bits":"0"*8})  # dummy bits
    # we consider FPR as fraction of samples that detector predicts any bit=1 above threshold 0.5
    # compute probs:
    model = DetectorNet(encoder, 8).to(device)
    model.load_state_dict(torch.load(model_path))
    cnt=0
    tot=0
    for h in human:
        toks = tokenizer(h["text"], return_tensors="pt", truncation=True, padding=True, max_length=192).to(device)
        with torch.no_grad():
            logits = model(input_ids=toks["input_ids"], attention_mask=toks["attention_mask"]).cpu().numpy()[0]
            probs = 1/(1+np.exp(-logits))
        if (probs>0.5).any():
            cnt+=1
        tot+=1
    fpr = cnt/tot
    results = {"orig":res_orig, "bt":res_bt, "fpr":fpr}
    with open(os.path.join(RESULTS,"phase0_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("Saved results to", os.path.join(RESULTS,"phase0_results.json"))
    print(results)

if __name__ == "__main__":
    main()