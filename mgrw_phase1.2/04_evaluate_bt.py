# evaluate_bt.py
import json
from decode_and_eval import decode_and_eval

texts_orig = json.load(open("data_texts.json"))
texts_bt   = json.load(open("data_texts_bt.json"))

results_orig = decode_and_eval(texts_orig)
results_bt   = decode_and_eval(texts_bt)

out = {
    "orig": results_orig,
    "bt": results_bt
}

json.dump(out, open("results_bt.json", "w"), indent=2)
print(json.dumps(out, indent=2))
