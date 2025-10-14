# src/train_injector_robust.py
"""
Robust Injector training:
- Model: mT5-small (PEFT LoRA)
- Loss: L = CE(reconstruct) + lambda_bit * BCE(bit_pred, bits) + lambda_sem * (1 - cos(orig_emb, wm_emb))
- Training data includes original and BT variants (same bits).
"""
import os, json
import torch
import torch.nn as nn
import numpy as np
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainer, Seq2SeqTrainingArguments
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sentence_transformers import SentenceTransformer, util
from torch.utils.data import DataLoader

# Paths
MODEL_NAME = "google/mt5-small"
PAIRS_TRAIN = "data/pairs/train.jsonl"
PAIRS_VAL = "data/pairs/val.jsonl"
OUTDIR = "outputs/injector_robust"

# Hyperparams
NUM_EPOCHS = 4
PER_DEVICE_BS = 4
GRAD_ACC = 8
LR = 1e-4
LAMBDA_BIT = 1.0
LAMBDA_SEM = 1.0
MAX_LEN = 192
LORA_R = 8

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=LORA_R,
    lora_alpha=16,
    target_modules=["q", "v", "k", "o"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, lora_config)

# sentence embedder (frozen) for semantic loss
embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)
embedder.eval()
for p in embedder.parameters():
    p.requires_grad = False

# small bit-predictor MLP that maps wm embedding -> bits
class BitPredictor(nn.Module):
    def __init__(self, emb_dim=384, nbits=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim, emb_dim//2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(emb_dim//2, nbits)
        )
    def forward(self, x):
        return self.net(x)

# load data
def load_jsonl(path):
    arr=[]
    with open(path, encoding="utf-8") as f:
        for l in f:
            arr.append(json.loads(l))
    return arr

train_items = load_jsonl(PAIRS_TRAIN)
val_items = load_jsonl(PAIRS_VAL)
nbits = len(train_items[0]["bits"])

# Dataset wrapper for tokenization
def preprocess_item(item):
    inp = f"watermark: {item['bits']} ||| {item['src']}"
    enc = tokenizer(inp, truncation=True, padding="max_length", max_length=MAX_LEN)
    with tokenizer.as_target_tokenizer():
        lab = tokenizer(item['src'], truncation=True, padding="max_length", max_length=MAX_LEN)
    enc['labels'] = lab['input_ids']
    enc['bits'] = item['bits']
    enc['orig_text'] = item['src']  # orig or bt variant
    return enc

train_ds = [preprocess_item(x) for x in train_items]
val_ds = [preprocess_item(x) for x in val_items]

# DataLoader (we will implement custom training loop to combine bit+semantic losses)
train_loader = DataLoader(train_ds, batch_size=PER_DEVICE_BS, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=PER_DEVICE_BS)

# Bit predictor attach
bit_predictor = BitPredictor(emb_dim=embedder.get_sentence_embedding_dimension(), nbits=nbits).to(device)
opt = torch.optim.AdamW(list(model.parameters()) + list(bit_predictor.parameters()), lr=LR, weight_decay=0.01)

# Training loop
ce_loss_f = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
bce_loss_f = nn.BCEWithLogitsLoss()

print("Start training on device:", device)
model.train()
for epoch in range(NUM_EPOCHS):
    tot_loss = 0.0
    for step, batch in enumerate(train_loader):
        # prepare inputs: collate manually
        inputs = tokenizer(batch['input_ids'] if isinstance(batch['input_ids'][0], list) else [tokenizer.decode(x) for x in batch['input_ids']],
                           return_tensors="pt", padding=True, truncation=True)
        # But above is messy; better to re-tokenize using stored strings:
        inputs_texts = [ ("watermark: {} ||| {}".format(b['bits'], b['orig_text'])) for b in batch ]
        targets = [ b['orig_text'] for b in batch ]
        enc = tokenizer(inputs_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(device)
        tgt = tokenizer(targets, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(device)
        labels = tgt["input_ids"].clone()
        labels[labels == tokenizer.pad_token_id] = -100

        outputs = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                        labels=labels, return_dict=True)
        loss_ce = outputs.loss  # seq2seq CE loss

        # generate wm text (greedy) for semantic & bit pred (we avoid expensive beam to save time)
        with torch.no_grad():
            generated = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                                       max_new_tokens=MAX_LEN, num_beams=2)
        gen_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)
        orig_texts = targets

        # semantic loss: 1 - cosine_sim(orig_emb, wm_emb)
        with torch.no_grad():
            orig_emb = embedder.encode(orig_texts, convert_to_tensor=True, show_progress_bar=False).to(device)
        wm_emb = embedder.encode(gen_texts, convert_to_tensor=True, show_progress_bar=False).to(device)
        cos_sim = torch.nn.functional.cosine_similarity(orig_emb, wm_emb, dim=-1)
        loss_sem = (1.0 - cos_sim).mean()

        # bit loss: predict bits from wm_emb
        # prepare bit labels
        bit_labels = torch.tensor([[int(ch) for ch in b['bits']] for b in batch], dtype=torch.float32).to(device)
        logits_bits = bit_predictor(wm_emb)
        loss_bit = bce_loss_f(logits_bits, bit_labels)

        loss = loss_ce + LAMBDA_BIT * loss_bit + LAMBDA_SEM * loss_sem

        loss.backward()
        if (step + 1) % GRAD_ACC == 0:
            opt.step()
            opt.zero_grad()

        tot_loss += loss.item()
        if step % 50 == 0:
            print(f"Epoch {epoch} step {step} loss {loss.item():.4f} (CE {loss_ce.item():.4f} bit {loss_bit.item():.4f} sem {loss_sem.item():.4f})")

    print(f"Epoch {epoch} avg loss {tot_loss / (step+1):.4f}")
    # save LoRA + bit predictor
    model.save_pretrained(OUTDIR)
    torch.save(bit_predictor.state_dict(), os.path.join(OUTDIR, "bit_predictor.pt"))
    print("Saved model to", OUTDIR)
