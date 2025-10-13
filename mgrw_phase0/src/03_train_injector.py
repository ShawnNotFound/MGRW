# src/03_train_injector.py
"""
Train a conditional paraphraser-style injector:
Input: "watermark: {bits} ||| {src}"
Target: src (reconstruction but conditioned on bits).
Model: google/mt5-small with PEFT LoRA
Saves to outputs/injector/
"""
import os, json
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments, Seq2SeqTrainer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

MODEL_NAME = "google/mt5-small"
OUT = "outputs/injector"
os.makedirs(OUT, exist_ok=True)
TRAIN_PATH = "data/pairs/train.jsonl"
VAL_PATH = "data/pairs/val.jsonl"

def load_jsonl(path):
    arr=[]
    with open(path, encoding="utf-8") as f:
        for l in f:
            arr.append(json.loads(l))
    return arr

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_item(x):
    inp = f"watermark: {x['bits']} ||| {x['src']}"
    enc = tokenizer(inp, truncation=True, padding="max_length", max_length=192)
    with tokenizer.as_target_tokenizer():
        lab = tokenizer(x['src'], truncation=True, padding="max_length", max_length=192)
    enc["labels"] = lab["input_ids"]
    return enc

train = load_jsonl(TRAIN_PATH)
val = load_jsonl(VAL_PATH)
train_ds = Dataset.from_list(train).map(lambda x: preprocess_item(x), remove_columns=list(train[0].keys()))
val_ds = Dataset.from_list(val).map(lambda x: preprocess_item(x), remove_columns=list(val[0].keys()))

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

training_args = Seq2SeqTrainingArguments(
    output_dir=OUT,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    predict_with_generate=True,
    fp16=True,
    gradient_accumulation_steps=8,
    num_train_epochs=4,
    logging_steps=50,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=1e-4,
    save_total_limit=2,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=tokenizer,
)

trainer.train()
trainer.save_model(OUT)
print("Saved injector to", OUT)