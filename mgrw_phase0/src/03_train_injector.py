# src/03_train_injector.py
"""
Train a conditional paraphraser-style injector:
Input: "watermark: {bits} ||| {src}"
Target: src (reconstruction but conditioned on bits).
Model: google/mt5-small with PEFT LoRA
Saves to outputs/injector/
"""
import os
import json
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# ---------- DEBUG SNIPPET (插入位置: 在 trainer = Seq2SeqTrainer(...) 之前或 map 完成后) ----------
import random
from torch.utils.data import DataLoader

def debug_dataset_sample(dataset, tokenizer, n_show=3, batch_size=8):
    print("\n=== Debug dataset sample & stats ===")
    # show first few tokenized samples
    for i in range(min(n_show, len(dataset))):
        ex = dataset[i]
        print(f"\n--- sample {i} raw keys: {list(ex.keys())} ---")
        # show decoded label (skip -100)
        lbl_ids = ex["labels"]
        # if labels are list of ints
        if isinstance(lbl_ids, list):
            clean = [tid for tid in lbl_ids if tid != -100 and tid != tokenizer.pad_token_id]
            print("decoded label:", tokenizer.decode(clean, skip_special_tokens=True))
        else:
            print("labels type:", type(lbl_ids))

    # create small dataloader with collator (if you have collator)
    try:
        coll = collator  # from your script
    except NameError:
        coll = None

    dl = DataLoader(dataset, batch_size=batch_size, collate_fn=coll if coll is not None else lambda x: x)
    # take one batch
    batch = next(iter(dl))
    # If collator used, batch will be dict with tensors
    print("\n--- batch inspection ---")
    if isinstance(batch, dict):
        input_ids = batch.get("input_ids")
        labels = batch.get("labels")
        print("batch input_ids shape:", getattr(input_ids, "shape", None))
        print("batch labels shape:", getattr(labels, "shape", None))
        # count -100 in labels
        lbl_arr = labels.cpu().numpy()
        total = lbl_arr.size
        neg100 = (lbl_arr == -100).sum()
        print(f"label -100 count: {neg100}/{total} ({neg100/total*100:.2f}%)")
        # show first example tokens / decoded
        first_labels = [int(x) for x in lbl_arr[0]]
        clean = [tid for tid in first_labels if tid != -100 and tid != tokenizer.pad_token_id]
        print("first example decoded labels:", tokenizer.decode(clean, skip_special_tokens=True))
    else:
        # no collator: batch is list of dicts
        print("batch is list; printing first element keys and label len")
        b0 = batch[0]
        print("keys:", list(b0.keys()))
        print("len labels:", len(b0.get("labels", [])))
    print("=== End debug ===\n")



# ---------------------
# Config
# ---------------------
MODEL_NAME = "google/mt5-small"
OUT = "outputs/injector"
os.makedirs(OUT, exist_ok=True)
TRAIN_PATH = "data/pairs/train.jsonl"
VAL_PATH = "data/pairs/val.jsonl"

# ---------------------
# Helpers
# ---------------------
def load_jsonl(path):
    arr = []
    with open(path, encoding="utf-8") as f:
        for l in f:
            arr.append(json.loads(l))
    return arr

# ---------------------
# Tokenizer + Preprocess
# ---------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_item(x):
    inp = f"watermark: {x['bits']} ||| {x['src']}"
    model_inputs = tokenizer(inp, truncation=True, padding="max_length", max_length=192)
    labels = tokenizer(
        x['src'],
        truncation=True,
        padding="max_length",
        max_length=192
    )["input_ids"]

    # 把 padding token 替换成 -100
    labels = [(-100 if t == tokenizer.pad_token_id else t) for t in labels]

    model_inputs["labels"] = labels
    return model_inputs



# ---------------------
# Load datasets
# ---------------------
train = load_jsonl(TRAIN_PATH)
val = load_jsonl(VAL_PATH)

cols_to_remove = ["src", "bits"]
train_ds = Dataset.from_list(train).map(lambda x: preprocess_item(x), remove_columns=cols_to_remove)
val_ds = Dataset.from_list(val).map(lambda x: preprocess_item(x), remove_columns=cols_to_remove)

# ---------------------
# Model + LoRA setup
# ---------------------
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q", "v", "k", "o"],
    lora_dropout=0.05,
    bias="none",
    task_type="SEQ_2_SEQ_LM"
)
model = get_peft_model(model, lora_config)

# ---------------------
# Data collator (⭐ prevents loss=0)
# ---------------------
collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=-100
)

# ---------------------
# Training arguments
# ---------------------
training_args = Seq2SeqTrainingArguments(
    output_dir=OUT,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    fp16=False,
    bf16=True,
    gradient_accumulation_steps=8,
    num_train_epochs=4,
    logging_steps=50,
    save_strategy="epoch",
    learning_rate=2e-5,  # ✅ safer LR
    save_total_limit=2,
    eval_strategy="epoch",  # ✅ works in transformers>=5
)

# ---------------------
# Trainer
# ---------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
    data_collator=collator
)

# ---------------------
# Train + save
# ---------------------

# print(tokenizer.decode([x for x in train_ds[0]["labels"] if x != -100]))
# print(tokenizer.decode(train_ds[0]["input_ids"]))

# debug_dataset_sample(train_ds, tokenizer, n_show=3, batch_size=8)

trainer.train()
trainer.save_model(OUT)
print("✅ Saved injector to", OUT)
