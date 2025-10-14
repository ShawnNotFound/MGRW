# data_prepare.py
from datasets import load_dataset
import random, json

dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
texts = [x["text"] for x in dataset if len(x["text"].split()) > 5]
texts = random.sample(texts, 200)

json.dump(texts, open("data_texts.json", "w"), indent=2)
print(f"Saved {len(texts)} samples to data_texts.json")
