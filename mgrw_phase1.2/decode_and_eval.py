# decode_and_eval.py
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import roc_auc_score
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

tok = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)

def get_embeddings(texts, batch_size=8):
    embs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            out = model(**inputs).last_hidden_state.mean(dim=1)
        embs.append(out.cpu())
    return torch.cat(embs, dim=0).numpy()

def decode_and_eval(texts):
    # 模拟标签：一半带水印，一半无水印
    n = len(texts)
    labels = np.array([1]*(n//2) + [0]*(n - n//2))
    embs = get_embeddings(texts)
    rng = np.random.default_rng(42)
    w = rng.normal(size=embs.shape[1])
    scores = embs.dot(w)

    auc = roc_auc_score(labels, scores)
    fpr = np.mean((scores > np.percentile(scores, 90)) & (labels == 0))
    brr = np.mean(scores[labels == 1] > np.median(scores))
    return {"brr": float(brr), "auc_mean": float(auc), "fpr": float(fpr)}
