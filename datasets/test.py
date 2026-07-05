import json

with open("gsm8k.json", "r", encoding="utf-8") as f:
    data = json.load(f)

print("Total questions:", len(data))

#  376 in truthful_qa.json
# 478 in strategyqa.json
# 500 in halueval_qa.json
# 500 in gsm8k.json