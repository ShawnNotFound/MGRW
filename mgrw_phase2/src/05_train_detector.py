# src/05_train_detector.py
"""
Train a detector that recovers bits from a single sentence.
Uses sentence-transformers all-MiniLM-L6-v2 as encoder and a small MLP.
Training data is the outputs from 04_generate_watermarked.py:
- outputs/watermarked/train_wm.jsonl (contains some BT variants if you added)
We create dataset of (text, bits) and train BCEWithLogitsLoss per bit.
Save to outputs/detector/
"""
import os, json
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
import numpy as np

WM_DIR = "outputs/watermarked"
OUT = "outputs/detector"
os.makedirs(OUT, exist_ok=True)
MODEL = "sentence-transformers/all-MiniLM-L6-v2"

tokenizer = AutoTokenizer.from_pretrained(MODEL)
encoder = AutoModel.from_pretrained(MODEL)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class WMDataset(Dataset):
    def __init__(self, jsonl_path):
        self.items = []
        with open(jsonl_path, encoding="utf-8") as f:
            for l in f:
                self.items.append(json.loads(l))
    def __len__(self): return len(self.items)
    def __getitem__(self,idx):
        txt = self.items[idx]["text"]
        bits = [int(ch) for ch in self.items[idx]["bits"]]
        return txt, np.array(bits, dtype=np.float32)

def collate_fn(batch):
    texts, bits = zip(*batch)
    toks = tokenizer(list(texts), padding=True, truncation=True, return_tensors="pt", max_length=192)
    bits = torch.tensor(np.stack(bits), dtype=torch.float32)
    return toks, bits

class Detector(nn.Module):
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

def main():
    train_ds = WMDataset(os.path.join(WM_DIR, "train_wm.jsonl"))
    val_ds = WMDataset(os.path.join(WM_DIR, "val_wm.jsonl"))
    nbits = len(train_ds[0][1])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=16, collate_fn=collate_fn)
    model = Detector(encoder, nbits).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    loss_f = nn.BCEWithLogitsLoss()
    for epoch in range(5):
        model.train()
        tot_loss = 0.0
        for toks, bits in tqdm(train_loader, desc=f"train ep{epoch}"):
            input_ids = toks["input_ids"].to(device)
            attn = toks["attention_mask"].to(device)
            labels = bits.to(device)
            logits = model(input_ids=input_ids, attention_mask=attn)
            loss = loss_f(logits, labels)
            loss.backward()
            optim.step()
            optim.zero_grad()
            tot_loss += loss.item()
        print(f"Epoch {epoch} loss {tot_loss/len(train_loader)}")
        # quick eval
        model.eval()
        all_y, all_p = [], []
        with torch.no_grad():
            for toks, bits in val_loader:
                input_ids = toks["input_ids"].to(device)
                attn = toks["attention_mask"].to(device)
                labels = bits.numpy()
                logits = model(input_ids=input_ids, attention_mask=attn).cpu().numpy()
                probs = 1/(1+np.exp(-logits))
                all_y.append(labels)
                all_p.append(probs)
        all_y = np.vstack(all_y); all_p = np.vstack(all_p)
        pred = (all_p>0.5).astype(int)
        brr = (pred == all_y).mean()
        print("Val BRR:", brr)
    torch.save(model.state_dict(), os.path.join(OUT, "detector.pt"))
    print("Saved detector to", OUT)

if __name__ == "__main__":
    main()