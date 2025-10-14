# backtranslate.py
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
import torch, json

def load_model(name):
    tok = MarianTokenizer.from_pretrained(name)
    mod = MarianMTModel.from_pretrained(name).to("cuda" if torch.cuda.is_available() else "cpu")
    return tok, mod

tok_en2fr, model_en2fr = load_model("Helsinki-NLP/opus-mt-en-fr")
tok_fr2en, model_fr2en = load_model("Helsinki-NLP/opus-mt-fr-en")

def translate(texts, tok, model, batch_size=4):
    results = []
    device = next(model.parameters()).device
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = texts[i:i+batch_size]
        inputs = tok(batch, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_length=256)
        results.extend(tok.batch_decode(outputs, skip_special_tokens=True))
    return results

texts = json.load(open("data_texts.json"))
texts_fr = translate(texts, tok_en2fr, model_en2fr)
texts_bt = translate(texts_fr, tok_fr2en, model_fr2en)

json.dump(texts_bt, open("data_texts_bt.json", "w"), indent=2)
print("Back-translation done → data_texts_bt.json")
